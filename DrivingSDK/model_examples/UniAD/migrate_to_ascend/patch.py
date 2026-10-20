# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# pylint: disable=too-many-lines
from __future__ import annotations

import importlib
import math
import os
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch_npu

from mx_driving.patcher import (
    AtomicPatch,
    BatchMatmul,
    CollectionsCompat,
    NpuFusedAdamwStep,
    Patch,
    Patcher,
    SigmoidFocalLoss,
    TensorIndex,
)
from mmdet.core import reduce_mean
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.models.builder import LOSSES
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (
    bbox_overlaps_nearest_3d as iou_3d,
)

if TYPE_CHECKING:
    from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin import Instances

# 移除上游同名注册，避免与 UniAD 工程内同名类重复注册
LOSSES.module_dict.pop("DiceLoss", None)
MATCH_COST.module_dict.pop("DiceCost", None)


def _set_hf32_flags() -> None:
    """开启 NPU 的 HF32计算开关。"""
    try:
        torch_npu.npu.matmul.allow_hf32 = True
        torch_npu.npu.conv.allow_hf32 = True
    except AttributeError:
        pass


def _matmul_contiguous(x, y):
    """matmul/bmm 前先做 contiguous（对应 tools/train.py 里的 bmm_custom）。"""
    return x.contiguous() @ y.contiguous()


# =============================================================================
# 补丁 1：引导期兼容 shims
# 对应 uniad.patch：
#   - datasets/eval_utils/map_api.py：删除 ``plt.style.use('seaborn-whitegrid')``
#   - dense_heads/occ_head_plugin/metrics.py：删除 IntersectionOverUnion 构造时
#     传入的 ``compute_on_step`` 参数（torchmetrics 新版本已移除该参数）
# =============================================================================


class UniADBootstrapCompat(Patch):
    """导入期兼容：吞掉已删除的 matplotlib style、剥离 compute_on_step 参数。"""

    name = "uniad_bootstrap_compat"
    apply_before_collect = True

    @staticmethod
    def _wrap_style_use(original):
        def style_use(style, *args, **kwargs):
            if style == "seaborn-whitegrid":
                return None
            return original(style, *args, **kwargs)

        return style_use

    @staticmethod
    def _wrap_metric_init(original):
        def metric_init(self, *args, **kwargs):
            kwargs.pop("compute_on_step", None)
            return original(self, *args, **kwargs)

        return metric_init

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            # matplotlib.style.use 与 .core.use 是两个独立绑定（core 在导入时被拷贝），
            # 必须两个都替换才能完整覆盖 map_api.py 里的调用。
            AtomicPatch(
                "matplotlib.style.core.use",
                target_wrapper=cls._wrap_style_use,
                aliases=["matplotlib.style.use"],
            ),
            AtomicPatch("torchmetrics.Metric.__init__", target_wrapper=cls._wrap_metric_init),
        ]


# =============================================================================
# 补丁 2：配置层
# =============================================================================


class UniADConfigPatch(Patch):
    """mmcv 加载配置后应用 UniAD NPU 训练默认值。"""

    name = "uniad_config"

    @staticmethod
    def _wrap_fromfile(original):
        def fromfile(filename, *args, **kwargs):
            cfg = original(filename, *args, **kwargs)
            base = os.path.basename(str(filename))
            if base not in {"base_track_map.py", "base_e2e.py"}:
                return cfg

            total_epochs = int(os.environ.get("UNIAD_TOTAL_EPOCHS", "4"))
            if hasattr(cfg, "optimizer"):
                cfg.optimizer["type"] = "NpuFusedAdamW"
            cfg.total_epochs = total_epochs
            if hasattr(cfg, "runner"):
                cfg.runner.max_epochs = total_epochs
            if hasattr(cfg, "test_pipeline"):
                cfg.evaluation = dict(interval=total_epochs, pipeline=cfg.test_pipeline)
            if hasattr(cfg, "log_config"):
                cfg.log_config.interval = int(os.environ.get("UNIAD_LOG_INTERVAL", "1"))
            return cfg

        return fromfile

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [AtomicPatch("mmcv.Config.fromfile", target_wrapper=cls._wrap_fromfile)]


# =============================================================================
# 补丁 3：bbox 工具
# denormalize_bbox 改为 split 实现并去掉 >8 维分支
# (vx/vy 始终输出，规避 NPU 上 split 取出的非连续切片问题)
# =============================================================================


class UniADBBox(Patch):
    """bbox 工具修复（split 版 denormalize_bbox，去掉 >8 分支）。"""

    name = "uniad_bbox"

    @staticmethod
    def _denormalize_bbox(normalized_bboxes, pc_range):
        cx, cy, w, length, cz, h, rot_sine, rot_cosine, vx, vy = normalized_bboxes.split(1, dim=-1)
        rot = torch.atan2(rot_sine, rot_cosine)
        w = w.exp()
        length = length.exp()
        h = h.exp()
        return torch.cat([cx, cy, cz, w, length, h, rot, vx, vy], dim=-1)

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch("projects.mmdet3d_plugin.core.bbox.util.denormalize_bbox", cls._denormalize_bbox),
            # track_loss 模块里用 ``from ....util import denormalize_bbox`` 按名引用，
            # 需要同时替换该模块内的名字绑定。
            AtomicPatch("projects.mmdet3d_plugin.losses.track_loss.denormalize_bbox", cls._denormalize_bbox),
        ]


# =============================================================================
# 补丁 4：开启DataLoader pin_memory=False → True
# （NPU 上与 H2D 流水配合更好）
# =============================================================================


def _worker_init_for_seed(worker_id, num_workers, rank, seed):
    """DataLoader worker 随机种子初始化。

    提为模块级函数：DataLoader 子进程在 spawn 启动方式下通过 pickle 传递
    worker_init_fn，局部函数（``<locals>``）无法 pickle 会报
    ``Can't pickle local object``；模块级函数可被 pickle，fork/spawn 通吃。
    语义与参考实现一致：worker_seed = num_workers * rank + worker_id + seed。
    """
    import random  # noqa: PLC0415

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UniADDataBuilder(Patch):
    """build_dataloader 重构（pin_memory=True 等 NPU 训练默认）。"""

    name = "uniad_data_builder"

    @staticmethod
    def _build_dataloader(
        dataset,
        samples_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=True,
        shuffle=True,
        seed=None,
        shuffler_sampler=None,
        nonshuffler_sampler=None,
        **kwargs,
    ):
        from functools import partial

        from mmcv.parallel import collate
        from mmcv.runner import get_dist_info
        from mmdet.datasets.samplers import GroupSampler
        from torch.utils.data import DataLoader

        from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

        rank, world_size = get_dist_info()
        if dist:
            if shuffle:
                sampler = build_sampler(
                    shuffler_sampler if shuffler_sampler is not None else dict(type="DistributedGroupSampler"),
                    dict(
                        dataset=dataset, samples_per_gpu=samples_per_gpu, num_replicas=world_size, rank=rank, seed=seed
                    ),
                )
            else:
                sampler = build_sampler(
                    nonshuffler_sampler if nonshuffler_sampler is not None else dict(type="DistributedSampler"),
                    dict(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed),
                )
            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

        init_fn = (
            partial(_worker_init_for_seed, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        )

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=True,
            worker_init_fn=init_fn,
            **kwargs,
        )

        return data_loader

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.datasets.builder.build_dataloader",
                cls._build_dataloader,
            ),
        ]


# =============================================================================
# 补丁 5：轨迹损失
# =============================================================================


class UniADTrajLoss(Patch):
    """轨迹评价指标强制 float（min_ade/min_fde/miss_rate）。"""

    name = "uniad_traj_loss"

    @staticmethod
    def _min_ade(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor):
        num_modes = traj.shape[1]
        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
        err = traj_gt_rpt - traj[:, :, :, 0:2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=3)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.clip(torch.sum((1 - masks_rpt), dim=2), min=1)
        err = err.float()
        err, inds = torch.min(err, dim=1)
        return err, inds

    @staticmethod
    def _min_fde(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor):
        num_modes = traj.shape[1]
        lengths = torch.sum(1 - masks, dim=1).long()
        valid_mask = lengths > 0
        traj = traj[valid_mask]
        traj_gt = traj_gt[valid_mask]
        masks = masks[valid_mask]
        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        lengths = torch.sum(1 - masks, dim=1).long()
        inds = lengths.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, num_modes, 1, 2) - 1
        traj_last = torch.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
        traj_gt_last = torch.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)
        err = traj_gt_last - traj_last[..., 0:2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=2)
        err = torch.pow(err, exponent=0.5)
        err = err.float()
        err, inds = torch.min(err, dim=1)
        return err, inds

    @staticmethod
    def _miss_rate(
        traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor, dist_thresh: float = 2
    ) -> torch.Tensor:
        num_modes = traj.shape[1]
        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
        dist = traj_gt_rpt - traj[:, :, :, 0:2]
        dist = torch.pow(dist, exponent=2)
        dist = torch.sum(dist, dim=3)
        dist = torch.pow(dist, exponent=0.5)
        dist[masks_rpt.bool()] = -math.inf
        dist = dist.float()
        dist, _ = torch.max(dist, dim=2)
        dist, _ = torch.min(dist, dim=1)
        m_r = torch.sum(torch.as_tensor(dist > dist_thresh)) / len(dist)
        return m_r

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        base = "projects.mmdet3d_plugin.losses.traj_loss"
        return [
            AtomicPatch(f"{base}.min_ade", cls._min_ade),
            AtomicPatch(f"{base}.min_fde", cls._min_fde),
            AtomicPatch(f"{base}.miss_rate", cls._miss_rate),
        ]


# =============================================================================
# 补丁 6：地图评估
# datasets/eval_utils/map_api.py：
#   - mask_for_polygons 处理 MultiPolygon（shapely 新版 geoms 属性）
#   - mask_for_lines 处理 MultiLineString
#   - 删除 ``plt.style.use('seaborn-whitegrid')``（由 UniADBootstrapCompat 处理）
# =============================================================================


class UniADMapApi(Patch):
    """Map 掩码绘制的 MultiPolygon / MultiLineString 兼容。"""

    name = "uniad_map_api"

    @staticmethod
    def _mask_for_polygons(polygons, mask):
        import cv2

        if not polygons:
            return mask

        def int_coords(x):
            return np.array(x).round().astype(np.int32)

        if hasattr(polygons, "geom_type") and polygons.geom_type == "MultiPolygon":
            polygons = list(polygons.geoms)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        cv2.fillPoly(mask, interiors, 0)
        return mask

    @staticmethod
    def _mask_for_lines(lines, mask):
        import cv2

        geometries = getattr(lines, "geoms", [lines])
        for geom in geometries:
            coords = np.asarray(list(geom.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)
        return mask

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        base = "projects.mmdet3d_plugin.datasets.eval_utils.map_api.NuScenesMapExplorer"
        # 上游属性是 @staticmethod（类级描述符）；保持该语义用 staticmethod 包裹，
        # 这样 ``self.mask_for_lines(a, b)`` 的实例调用不会绑定 self。
        return [
            AtomicPatch(f"{base}.mask_for_polygons", staticmethod(cls._mask_for_polygons)),
            AtomicPatch(f"{base}.mask_for_lines", staticmethod(cls._mask_for_lines)),
        ]


# =============================================================================
# 补丁 7：数据 pipeline
# datasets/pipelines/loading.py：去掉 ``img_root`` 前缀拼接
# （数据走 Ceph 全路径），运行时等价做法是加载期间把 img_root 置空
# =============================================================================


class UniADDataPipeline(Patch):
    """LoadMultiViewImageFromFilesInCeph 加载期间保持 Ceph 路径原样。"""

    name = "uniad_data_pipeline"

    @staticmethod
    def _wrap_load_multiview(original):
        def load_multiview(self, results):
            old_img_root = getattr(self, "img_root", None)
            if old_img_root is not None:
                self.img_root = ""
            try:
                return original(self, results)
            finally:
                if old_img_root is not None:
                    self.img_root = old_img_root

        return load_multiview

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.datasets.pipelines.loading.LoadMultiViewImageFromFilesInCeph.__call__",
                target_wrapper=cls._wrap_load_multiview,
            ),
        ]


# =============================================================================
# 补丁 8：MSDA 算子
# 四处 MSDA 调用点替换（motion_deformable_attn.py /
# decoder.py / temporal_self_attention.py / spatial_cross_attention.py 的
# ``MultiScaleDeformableAttnFunction.apply`` → ``mx_driving.fused.xxx``）。
# 这四处调用最终都落入 UniAD 工程 vendored 的 multi_scale_deformable_attn_function，
# 运行时把该 autograd 的 forward/backward 重定向到 mx_driving 实现即可全覆盖，
# 并保留原有 autograd 可导性。
# =============================================================================


class UniADLocalMSDA(Patch):
    """将 UniAD vendored 的 MSDA autograd 前后向重定向到 mx_driving。"""

    name = "uniad_local_msda"

    @staticmethod
    def _forward_wrapper(npu_forward):
        def forward(
            ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=None
        ):
            # mx_driving 实现不接收 im2col_step，丢弃该参数即可
            return npu_forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

        return forward

    @staticmethod
    def _backward_wrapper(npu_backward):
        def backward(ctx, grad_output):
            # mx_driving backward 返回 5 个梯度，补齐 im2col_step 对应的 None
            return (*npu_backward(ctx, grad_output), None)

        return backward

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        module = "projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function"
        bases = [
            f"{module}.MultiScaleDeformableAttnFunction_fp32",
            f"{module}.MultiScaleDeformableAttnFunction_fp16",
        ]
        patches = []
        for base in bases:
            patches.extend(
                [
                    AtomicPatch(
                        f"{base}.forward",
                        "mx_driving.MultiScaleDeformableAttnFunction.forward",
                        replacement_wrapper=cls._forward_wrapper,
                    ),
                    AtomicPatch(
                        f"{base}.backward",
                        "mx_driving.MultiScaleDeformableAttnFunction.backward",
                        replacement_wrapper=cls._backward_wrapper,
                    ),
                ]
            )
        return patches


# =============================================================================
# 补丁 9：功能函数
# models/utils/functional.py：anchor_coordinate_transform
# 中 matmul 之前先对算子做广播展开（避免非连续/广播歧义）
# =============================================================================


class UniADFunctional(Patch):
    """anchor_coordinate_transform 广播展开。"""

    name = "uniad_functional"

    @staticmethod
    def _anchor_coordinate_transform(
        anchors,
        bbox_results,
        with_translation_transform=True,
        with_rotation_transform=True,
    ):
        module = importlib.import_module("projects.mmdet3d_plugin.models.utils.functional")
        rot_2d = module.rot_2d
        rearrange = module.rearrange

        batch_size = len(bbox_results)
        batched_anchors = []
        transformed_anchors = anchors[None, ...]
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            yaw = bboxes.yaw.to(transformed_anchors.device)
            bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
            if with_rotation_transform:
                angle = yaw - 3.1415953
                rot_yaw = rot_2d(angle)
                rot_yaw = rot_yaw[:, None, None, :, :]
                transformed_anchors = rearrange(transformed_anchors, "b g m t c -> b g m c t")
                num_agents, _, _, _, _ = rot_yaw.shape
                _, num_groups, num_modes, _, _ = transformed_anchors.shape
                rot_yaw = rot_yaw.expand((num_agents, num_groups, num_modes, 2, 2))
                transformed_anchors = transformed_anchors.expand((num_agents, num_groups, num_modes, 2, 12))
                transformed_anchors = torch.matmul(rot_yaw, transformed_anchors)
                transformed_anchors = rearrange(transformed_anchors, "b g m c t -> b g m t c")
            if with_translation_transform:
                transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
            batched_anchors.append(transformed_anchors)
        return torch.stack(batched_anchors)

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.utils.functional.anchor_coordinate_transform",
                cls._anchor_coordinate_transform,
            ),
        ]


# =============================================================================
# 补丁 10：训练/测试并行封装
# uniad/apis/mmdet_train.py 与 tools/test.py：
# MMDataParallel → NPUDataParallel、MMDistributedDataParallel →
# NPUDistributedDataParallel
# =============================================================================


class UniADTraining(Patch):
    """把训练/测试用到的 mmcv 并行封装替换为 NPU 版本。"""

    name = "uniad_training"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.apis.mmdet_train.MMDataParallel",
                "mmcv.device.npu.NPUDataParallel",
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.apis.mmdet_train.MMDistributedDataParallel",
                "mmcv.device.npu.NPUDistributedDataParallel",
            ),
            AtomicPatch("tools.test.MMDataParallel", "mmcv.device.npu.NPUDataParallel"),
            AtomicPatch("tools.test.MMDistributedDataParallel", "mmcv.device.npu.NPUDistributedDataParallel"),
        ]


# =============================================================================
# 补丁 11：MotionHead 索引
# dense_heads/motion_head.py：from_track 里
# ``matched_gt_idx = matched_gt_idx.detach().cpu()``（高级索引的 index 放 CPU）
# =============================================================================


class UniADMotionHead(Patch):
    """MotionHead 计算 matched gt 轨迹前先把索引 detach 到 CPU。"""

    name = "uniad_motion_head"

    @staticmethod
    def _wrap_compute_matched_gt_traj(original):
        def compute_matched_gt_traj(
            self,
            gt_fut_traj,
            gt_fut_traj_mask,
            all_matched_idxes,
            track_bbox_results,
            gt_bboxes_3d,
        ):
            fixed_idxes = [idx.detach().cpu() if hasattr(idx, "detach") else idx for idx in all_matched_idxes]
            return original(
                self,
                gt_fut_traj,
                gt_fut_traj_mask,
                fixed_idxes,
                track_bbox_results,
                gt_bboxes_3d,
            )

        return compute_matched_gt_traj

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.dense_heads.motion_head.MotionHead.compute_matched_gt_traj",
                target_wrapper=cls._wrap_compute_matched_gt_traj,
            ),
        ]


# =============================================================================
# 补丁 12：规划碰撞指标
# planning_head_plugin/planning_metrics.py：
# evaluate_coll 中 yi/xi/m2 等高级索引张量先 .cpu()（避免 NPU 上 ./ 后
# 非整型索引报错）
# =============================================================================


class UniADPlanningMetric(Patch):
    """规划碰撞评估的索引张量移到 CPU（等效源码侧 .cpu() 改动）。"""

    name = "uniad_planning_metric"

    @staticmethod
    def _evaluate_coll(self, trajs, gt_trajs, segmentation):  # pylint: disable=bad-staticmethod-argument
        batch, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)
        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(batch):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])
            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()
            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))
            ti = torch.arange(n_future)
            yi = yi.detach().cpu()
            xi = xi.detach().cpu()
            m1 = m1.detach().cpu()
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            m2 = torch.logical_not(gt_box_coll).detach().cpu()
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].long()
        return obj_coll_sum, obj_box_coll_sum

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin.planning_metrics.PlanningMetric.evaluate_coll",
                cls._evaluate_coll,
            ),
        ]


# =============================================================================
# 补丁 13：检测器 loss 归一化
# uniad/detectors/uniad_e2e.py：
# ``for k,v in losses.items(): v = v.float(); losses[k] = torch.nan_to_num(v)``。
# 运行时等价值：在 original 执行期间临时安装"宽容版 nan_to_num"（double→float
# 后转发），同时覆盖 forward_train 内部所有源码侧调用点（uniad_e2e/track_head
# 等），语义一致（int/bool 本就在 aclnnNanToNum 支持列表内）。
# =============================================================================


class UniADDetector(Patch):
    """loss 归一化前强制 float（double → float 再 nan_to_num）。"""

    name = "uniad_detector"

    @staticmethod
    def _wrap_forward_train(original):
        def forward_train(self, *args, **kwargs):  # pylint: disable=redefined-outer-name
            _orig = torch.nan_to_num

            def _safe(v, *a, **kw):
                if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                    v = v.float()
                return _orig(v, *a, **kw)

            torch.nan_to_num = _safe
            try:
                return original(self, *args, **kwargs)
            finally:
                torch.nan_to_num = _orig

        return forward_train

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.detectors.uniad_e2e.UniAD.forward_train",
                target_wrapper=cls._wrap_forward_train,
            ),
        ]


# =============================================================================
# 补丁 14：Encoder 点采样
# uniad/modules/encoder.py：point_sampling 中
# ``torch.matmul(lidar2img, reference_points)`` 改为
# ``torch.mul(...).transpose(-1,-2).sum(-1, keepdim=True).squeeze(-1)``
# （避免 NPU matmul 的中间非连续布局）。
# =============================================================================


class UniADEncoder(Patch):
    """NPU 友好的 BEVFormer point_sampling（matmul→mul+sum）。"""

    name = "uniad_encoder"

    @staticmethod
    def _point_sampling(self, reference_points, pc_range, img_metas):  # pylint: disable=bad-staticmethod-argument
        from mmcv.utils import TORCH_VERSION, digit_version

        def _impl(reference_points, pc_range, img_metas):
            reference_points = reference_points.float()
            lidar2img = []
            for img_meta in img_metas:
                lidar2img.append(img_meta["lidar2img"])
            lidar2img = np.asarray(lidar2img)
            lidar2img = reference_points.new_tensor(lidar2img)
            reference_points = reference_points.clone()
            reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
            reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
            reference_points = reference_points.permute(1, 0, 2, 3)
            depth, batch, num_query = reference_points.size()[:3]
            num_cam = lidar2img.size(1)
            reference_points = (
                reference_points.view(depth, batch, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
            )
            lidar2img = lidar2img.view(1, batch, num_cam, 1, 4, 4).repeat(depth, 1, 1, num_query, 1, 1)
            reference_points_cam = (
                torch.mul(
                    lidar2img.to(torch.float32),
                    reference_points.to(torch.float32).transpose(-1, -2),
                )
                .sum(-1, keepdim=True)
                .squeeze(-1)
            )
            eps = 1e-5
            bev_mask = reference_points_cam[..., 2:3] > eps
            reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
                reference_points_cam[..., 2:3],
                torch.ones_like(reference_points_cam[..., 2:3]) * eps,
            )
            reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
            reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]
            bev_mask = (
                bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0)
            )
            if digit_version(TORCH_VERSION) >= digit_version("1.8"):
                bev_mask = torch.nan_to_num(bev_mask)
            else:
                bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))
            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
            bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
            return reference_points_cam, bev_mask

        return _impl(reference_points, pc_range, img_metas)

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.modules.encoder.BEVFormerEncoder.point_sampling",
                cls._point_sampling,
            ),
        ]


# =============================================================================
# 补丁 15：SCA 索引缓存
# uniad/modules/spatial_cross_attention.py：
# forward 中把按 bev_mask 逐点计算的 indexes/max_len/count 缓存起来
# （同一 bev_mask 张量跨 iter 复用，避免反复 reduce/permute）
# =============================================================================


class UniADSpatialCrossAttention(Patch):
    """SpatialCrossAttention.forward 缓存 BEV mask 索引，避免重复归约。"""

    name = "uniad_spatial_cross_attention"
    _indexes = None
    _max_len = None
    _bev_mask_id = -1
    _count = None

    @staticmethod
    def _forward(  # pylint: disable=bad-staticmethod-argument
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        else:
            inp_residual = residual
            slots = torch.zeros_like(residual)
        if query_pos is not None:
            query = query + query_pos

        batch, _, _ = query.size()
        depth = reference_points_cam.size(3)
        bev_mask_id = id(bev_mask)
        if bev_mask_id == UniADSpatialCrossAttention._bev_mask_id:
            indexes = UniADSpatialCrossAttention._indexes
            max_len = UniADSpatialCrossAttention._max_len
            count = UniADSpatialCrossAttention._count
        else:
            count = torch.any(bev_mask, 3)
            bev_mask_compact = count.squeeze()
            indexes = []
            for mask_per_img in bev_mask_compact:
                indexes.append(mask_per_img.nonzero().squeeze(-1))
            max_len = max(len(each) for each in indexes)
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            count = count[..., None]
            UniADSpatialCrossAttention._indexes = indexes
            UniADSpatialCrossAttention._max_len = max_len
            UniADSpatialCrossAttention._bev_mask_id = bev_mask_id
            UniADSpatialCrossAttention._count = count

        queries_rebatch = query.new_zeros([batch, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([batch, self.num_cams, max_len, depth, 2])
        for i, reference_points_per_img in enumerate(reference_points_cam):
            index_query_per_img = indexes[i]
            for j in range(batch):
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        num_cams, length, batch, _ = key.shape
        key = key.permute(2, 0, 1, 3).reshape(batch * self.num_cams, length, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(batch * self.num_cams, length, self.embed_dims)
        queries = self.deformable_attention(
            query=queries_rebatch.view(batch * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(batch * self.num_cams, max_len, depth, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(batch, self.num_cams, max_len, self.embed_dims)

        for j in range(batch):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]
        slots = slots / count
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.modules.spatial_cross_attention.SpatialCrossAttention.forward",
                cls._forward,
            ),
        ]


# =============================================================================
# 补丁 16：BEV 旋转
# uniad/modules/transformer.py：
# ``F.rotate(tmp_prev_bev, angle)`` → 显式 ``InterpolationMode.BILINEAR``
# =============================================================================


class UniADTransformer(Patch):
    """前一帧 BEV 旋转使用 BILINEAR 插值。"""

    name = "uniad_transformer"

    @staticmethod
    def _rotate(img, angle, interpolation=None, expand=False, center=None, fill=None):
        from torchvision.transforms.functional import InterpolationMode, rotate

        return rotate(img, angle, InterpolationMode.BILINEAR, expand=expand, center=center, fill=fill)

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch("projects.mmdet3d_plugin.uniad.modules.transformer.rotate", cls._rotate),
        ]


# ---------------------------------------------------------------------------
# 1) 并行 + 异步 ``nonlinear_smoother``
#
# dense_heads/motion_head_plugin/motion_utils.py：
# 「按 agent 串行 casadi/IPOPT 求解」→「多进程 pool 求解」（每个 worker 按
# 轨迹长度缓存一份 Opti 实例）。门控条件、求解器设置与输出拼装与上游串行
# 实现完全一致，输出逐位一致。
# ---------------------------------------------------------------------------

_SMOOTHER_WORKERS = int(os.environ.get("UNIAD_SMOOTHER_WORKERS", "16"))


class _SmootherPool:
    """按需创建的多进程求解池（每个 agent 的 casadi 求解）。"""

    _pool = None
    _ctx = None

    @classmethod
    def _ensure_pool(cls):
        if cls._pool is None:
            import multiprocessing as mp

            # fork：worker 继承父进程 python 镜像（COW），只跑纯 numpy/casadi 代码，
            # 不接触 torch/NPU 句柄。
            ctx = mp.get_context("fork")
            cls._ctx = ctx
            cls._pool = ctx.Pool(
                processes=max(1, min(_SMOOTHER_WORKERS, os.cpu_count() or 8)),
                initializer=_init_worker,
            )
        return cls._pool

    @classmethod
    def solve_batch(cls, jobs):
        """jobs: list of (ts, x_curr, ref)。返回 list of ndarray (N+1,2)。"""
        if not jobs:
            return []
        pool = cls._ensure_pool()
        return pool.map(_solve_job, jobs)

    @classmethod
    def solve_batch_async(cls, jobs):
        """非阻塞提交；结果对象 .get() 返回 list of ndarray (N+1,2)。"""
        if not jobs:
            return _EmptyAsyncResult()
        pool = cls._ensure_pool()
        return pool.map_async(_solve_job, jobs)


class _EmptyAsyncResult:
    """空任务列表的立即结果。"""

    def get(self, timeout=None):
        return []


def _init_worker():
    """每个 worker 进程初始化一次：按 ts 构建 Opti 实例并缓存。"""
    import threading
    from projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.motion_optimization import MotionNonlinearSmoother

    _init_worker.cache = {}
    _init_worker.lock = threading.Lock()
    _init_worker.cls = MotionNonlinearSmoother


def _get_smoother(ts):
    cache = getattr(_init_worker, "cache", None)
    if cache is None:
        _init_worker()
        cache = _init_worker.cache
    sm = cache.get(ts)
    if sm is None:
        sm = _init_worker.cls(trajectory_len=ts, dt=0.5)
        cache[ts] = sm
    return sm


def _solve_job(job):
    """在 worker 进程内求解一个平滑问题。"""
    ts, x_curr, ref = job
    sm = _get_smoother(ts)
    sm.set_reference_trajectory(x_curr, ref[: ts + 1, :])
    sol = sm.solve()
    return np.stack([sol.value(sm.position_x), sol.value(sm.position_y)], axis=-1)


def _prepare(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor):
    """输入转 numpy 并推导每个 agent 的元信息（与原实现一致）。"""
    device = gt_fut_traj.device
    dtype = gt_fut_traj.dtype
    gt_bboxes_3d = gt_bboxes_3d.cpu().detach().numpy()
    gt_fut_traj = gt_fut_traj.cpu().detach().numpy()
    gt_fut_traj_xy_diff = np.zeros((gt_fut_traj.shape[0], 13, 2))
    gt_fut_traj_xy_diff[:, 1:, :] = gt_fut_traj
    gt_fut_traj_xy_diff = np.diff(gt_fut_traj_xy_diff, axis=1)
    gt_fut_traj_yaw = np.arctan2(gt_fut_traj_xy_diff[:, :, 1], gt_fut_traj_xy_diff[:, :, 0])
    gt_fut_traj_yaw = np.concatenate([-np.pi / 2 - gt_bboxes_3d[:, None, 6:7], gt_fut_traj_yaw[:, :, None]], axis=1)
    gt_fut_traj = np.concatenate([gt_bboxes_3d[:, None, :2], gt_fut_traj], axis=1)

    gt_fut_traj_mask = gt_fut_traj_mask.cpu().detach().numpy()
    bbox_tensor = bbox_tensor.cpu().detach().numpy()
    ts_limit = gt_fut_traj_mask.sum(1)[:, 0]
    yaw_preds = bbox_tensor[:, 6]
    vel_preds = bbox_tensor[:, -2:]
    speed_preds = np.sqrt(np.sum(vel_preds**2, axis=-1))
    return dict(
        device=device,
        dtype=dtype,
        gt_fut_traj=gt_fut_traj,
        gt_fut_traj_yaw=gt_fut_traj_yaw,
        gt_fut_traj_mask=gt_fut_traj_mask,
        ts_limit=ts_limit,
        bbox_tensor=bbox_tensor,
        yaw_preds=yaw_preds,
        speed_preds=speed_preds,
    )


def _is_dynamic(traj, ts, dist_thres):
    return np.sqrt(np.sum((traj[ts, :2] - traj[0, :2]) ** 2)) > dist_thres


def _check_diff(x_curr, ref_traj):
    if np.sqrt((x_curr[0] - ref_traj[0, 0]) ** 2 + (x_curr[1] - ref_traj[0, 1]) ** 2) > 2:
        return False
    a = np.array([np.cos(x_curr[2]), np.sin(x_curr[2])])
    b = np.array([np.cos(ref_traj[0, 2]), np.sin(ref_traj[0, 2])])
    diff_theta = np.arccos(np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))))
    if diff_theta > np.pi / 180 * 30:
        return False
    return True


def _check_ade(traj_pert, traj_ref, thres):
    return np.mean(np.sqrt(np.sum((traj_pert[:, :2] - traj_ref[:, :2]) ** 2, axis=-1))) < thres


def _build_jobs(prep):
    """门控循环 → (jobs, solve_indices)。与原实现一致。"""
    gt_fut_traj = prep["gt_fut_traj"]
    gt_fut_traj_yaw = prep["gt_fut_traj_yaw"]
    ts_limit = prep["ts_limit"]
    bbox_tensor = prep["bbox_tensor"]
    yaw_preds = prep["yaw_preds"]
    speed_preds = prep["speed_preds"]
    solve_jobs = []
    solve_indices = []
    for i in range(gt_fut_traj.shape[0]):
        ts = ts_limit[i]
        x_curr = [bbox_tensor[i, 0], bbox_tensor[i, 1], -np.pi / 2 - yaw_preds[i], speed_preds[i]]
        reference_trajectory = np.concatenate([gt_fut_traj[i], gt_fut_traj_yaw[i]], axis=-1)
        if ts > 1 and _is_dynamic(gt_fut_traj[i], int(ts), 2) and _check_diff(x_curr, reference_trajectory):
            solve_jobs.append((int(ts), x_curr, reference_trajectory))
            solve_indices.append(i)
    return solve_jobs, solve_indices


def _assemble(prep, solved, solve_indices):
    """拼装 traj_perturb_all + mask 张量。与原实现一致。"""
    device = prep["device"]
    dtype = prep["dtype"]
    gt_fut_traj = prep["gt_fut_traj"]
    gt_fut_traj_yaw = prep["gt_fut_traj_yaw"]
    gt_fut_traj_mask = prep["gt_fut_traj_mask"]
    ts_limit = prep["ts_limit"]
    traj_perturb_all = []
    solve_indices = list(solve_indices)
    solved_iter = iter(solved)
    for i in range(gt_fut_traj.shape[0]):
        ts = ts_limit[i]
        reference_trajectory = np.concatenate([gt_fut_traj[i], gt_fut_traj_yaw[i]], axis=-1)
        if len(solve_indices) and i == solve_indices[0]:
            solve_indices.pop(0)
            traj_perturb = next(solved_iter)
            if not _check_ade(traj_perturb, reference_trajectory[: int(ts) + 1], thres=1.5):
                traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]
            else:
                traj_perturb_tmp = traj_perturb[1:, :2] - traj_perturb[0:1, :2]
                traj_perturb = np.zeros((12, 2))
                traj_perturb[: traj_perturb_tmp.shape[0], :] = traj_perturb_tmp[:, :2]
        else:
            traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]
        traj_perturb_all.append(traj_perturb)
    return torch.tensor(traj_perturb_all, device=device, dtype=dtype), torch.tensor(gt_fut_traj_mask > 0, device=device)


def nonlinear_smoother(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor):
    """（并行版）对 GT 未来轨迹施加非线性平滑。

    Args:
        gt_bboxes_3d (torch.Tensor): (batch_size, 7) 的 3D 框。
        gt_fut_traj (torch.Tensor): (batch_size, 12, 2) 的未来轨迹。
        gt_fut_traj_mask (torch.Tensor): (batch_size, 12) 有效时间步掩码。
        bbox_tensor (torch.Tensor): (batch_size, 9) 的框属性。

    Returns:
        torch.Tensor: (batch_size, 12, 2) 扰动轨迹。
        torch.Tensor: 与 gt_fut_traj_mask 同形的有效掩码。
    """
    prep = _prepare(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor)
    solve_jobs, solve_indices = _build_jobs(prep)
    solved = _SmootherPool.solve_batch(solve_jobs) if solve_jobs else []
    return _assemble(prep, solved, solve_indices)


class SmootherAsyncResult:
    """异步提交的 smoother 批次的挂起结果。

    `.get()` 会阻塞直到 casadi 求解完成，返回与 `nonlinear_smoother`
    相同的 (traj, mask) 张量。
    """

    def __init__(self, prep, solve_indices, async_result):
        self._prep = prep
        self._solve_indices = solve_indices
        self._async_result = async_result

    def get(self, timeout=None):
        solved = self._async_result.get(timeout=timeout)
        return _assemble(self._prep, solved, self._solve_indices)


def submit_nonlinear_smoother(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor):
    """`nonlinear_smoother` 的非阻塞版本。

    返回 `SmootherAsyncResult`；调用 `.get()` 得到与同步函数相同的
    (traj, mask) 张量。
    """
    prep = _prepare(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor)
    solve_jobs, solve_indices = _build_jobs(prep)
    async_result = _SmootherPool.solve_batch_async(solve_jobs)
    return SmootherAsyncResult(prep, solve_indices, async_result)


# ---------------------------------------------------------------------------
# 2) 矢量化的 ``OccHead.forward_train``
#
# dense_heads/occ_head.py：原来的实现遍历 query 并在循环
# 内调用 ``.item()``，造成逐点 host-device 同步；改为 device 侧矢量化
# （按 chunk 分批以控制中间显存），4/4 等价性用例验证过。
# ---------------------------------------------------------------------------


def _build_ins_seg_targets_ordered(ins_seg_targets, cur_gt_inds, ignore_index):
    # 分配 gt 的矢量化实现（去掉逐实例 .item()/device 同步）
    # -1 表示未匹配 query；-2 表示 no_query；ignore_index(255) 特殊目标
    # → 映射为 -20（与 occ_label.py 一致）
    SPECIAL_INDEX = -20
    n_query = cur_gt_inds.shape[0]
    all_ignore = (ins_seg_targets == ignore_index).all()
    if all_ignore.item() is True:
        # 直接对每个 query 使用 occ-and-flow gt
        ins_seg_targets_ordered = ins_seg_targets.long().unsqueeze(0).expand(n_query, -1, -1, -1).contiguous()
    else:
        pos_query = (cur_gt_inds != -1) & (cur_gt_inds != -2)  # (Q,)
        ids_safe = torch.where(pos_query, cur_gt_inds, torch.zeros_like(cur_gt_inds))
        ids_safe = torch.where(ids_safe == ignore_index, torch.full_like(ids_safe, SPECIAL_INDEX), ids_safe)
        # 分 chunk 以限制 (Q,t,h,w) 中间张量的峰值显存
        ordered_chunks = []
        _chunk = 64
        for s in range(0, n_query, _chunk):
            ids_chunk = ids_safe[s : s + _chunk].view(-1, 1, 1, 1)
            pos_chunk = pos_query[s : s + _chunk].view(-1, 1, 1, 1)
            cur = (ins_seg_targets.unsqueeze(0) == ids_chunk).long()
            cur = torch.where(pos_chunk, cur, torch.ones_like(cur) * ignore_index)
            ordered_chunks.append(cur)
        ins_seg_targets_ordered = torch.cat(ordered_chunks, dim=0)  # [n_gt, t, h, w]
    return ins_seg_targets_ordered


def forward_train(
    self,
    bev_feat,
    outs_dict,
    gt_inds_list=None,
    gt_segmentation=None,
    gt_instance=None,
    gt_img_is_valid=None,
):
    # 生成 warp 后的 gt 及相关输入
    gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)

    all_matched_gt_ids = outs_dict['all_matched_idxes']  # list of tensor, length bs

    ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

    # 前向 occ-flow 模型
    mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)

    # pred 与 gt
    ins_seg_targets_batch = gt_instance  # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch

    # img_valid 标志：序列中无效样本在计算 loss 时过滤
    img_is_valid = gt_img_is_valid  # [1, 7]
    assert img_is_valid.size(1) == self.receptive_field + self.n_future, (
        "Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
    )
    frame_valid_mask = img_is_valid.bool()
    past_valid_mask = frame_valid_mask[:, : self.receptive_field]
    future_frame_mask = frame_valid_mask[:, (self.receptive_field - 1) :]  # [1, 5]  including current frame

    # 仅当过去 3 帧全部有效时才监督
    past_valid = past_valid_mask.all(dim=1)
    future_frame_mask[~past_valid] = False

    # 计算 batch 内 loss
    loss_dict = dict()
    loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
    loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
    loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
    loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

    bs = ins_query.size(0)
    assert bs == 1
    for ind in range(bs):
        # 每个 gt_bboxes 含 3 帧，只用最后一帧
        cur_gt_inds = gt_inds_list[ind][-1]

        cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt]

        # 按 matched_gt_inds 重排 gt
        cur_gt_inds = cur_gt_inds[cur_matched_gt]

        # matched_gt=-1 实际上是背景（未匹配）
        cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
        cur_gt_inds[cur_matched_gt == -2] = -2

        frame_mask = future_frame_mask[ind]  # [t]

        # 预测
        ins_seg_preds = ins_seg_preds_batch[ind]  # [q(n_gt for matched), t, h, w]
        ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
        mask_preds = mask_preds_batch[ind]
        ins_seg_targets_ordered = _build_ins_seg_targets_ordered(ins_seg_targets, cur_gt_inds, self.ignore_index)

        # 形状校验
        t, h, w = ins_seg_preds.shape[-3:]
        assert t == 1 + self.n_future, f"{ins_seg_preds.size()}"
        assert ins_seg_preds.size() == ins_seg_targets_ordered.size(), (
            f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
        )

        num_total_pos = ins_seg_preds.size(0)

        # batch 内单个样本的 loss
        num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        cur_dice_loss = self.loss_dice(
            ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask
        )

        cur_mask_loss = self.loss_mask(ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask)

        cur_aux_dice_loss = self.loss_dice(
            mask_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask
        )
        cur_aux_mask_loss = self.loss_mask(mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask)

        loss_dice += cur_dice_loss
        loss_mask += cur_mask_loss
        loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
        loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

    loss_dict['loss_dice'] = loss_dice / bs
    loss_dict['loss_mask'] = loss_mask / bs
    loss_dict['loss_aux_dice'] = loss_aux_dice / bs
    loss_dict['loss_aux_mask'] = loss_aux_mask / bs

    return loss_dict


# ---------------------------------------------------------------------------
# 3) loss 侧优化
#
# * ``CollisionLoss.forward``（planning_loss.py hunk）—— 把逐目标框的
#   相交计算循环矢量化，避免逐框 D2H/H2D；
# * ``seg_utils.IOU``（seg_utils.py hunk）—— 去掉末尾 ``.cpu()``（D2H）；
# * ``ClipMatcher``（track_loss.py hunk）—— 把逐 track obj-id 匹配循环
#   矢量化（match_for_single_frame / prediction_loss），并让 loss_boxes
#   的 keep 过滤在全部保留时变成 no-op。
# ---------------------------------------------------------------------------


def collision_loss_forward_vectorized(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
    # sdc_traj_all (1, 6, 2)
    # sdc_planning_gt (1,6,3)
    # sdc_planning_gt_mask (1, 6)
    # future_gt_bbox 6x[lidarboxinstance]
    n_futures = len(future_gt_bbox)
    inter_sum = torch.tensor([0.0], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    for i in range(n_futures):
        if len(future_gt_bbox[i].tensor) > 0:
            future_gt_corners = (
                future_gt_bbox[i].corners[:, [0, 3, 4, 7], :2].to(sdc_traj_all.device)
            )  # (N, 8, 3) -> (N, 4, 2) only bev
            sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
            sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
            sdc_min = sdc_bev_box.min(dim=0)[0]  # (2,)
            sdc_max = sdc_bev_box.max(dim=0)[0]  # (2,)

            # 一次算完该帧所有目标框的 min/max（矢量化）
            target_min = future_gt_corners.min(dim=1)[0]  # (N, 2)
            target_max = future_gt_corners.max(dim=1)[0]  # (N, 2)

            # 对全部框计算相交（矢量化）
            intersect_min = torch.maximum(sdc_min, target_min)  # (N, 2)
            intersect_max = torch.minimum(sdc_max, target_max)  # (N, 2)
            intersect_dims = intersect_max - intersect_min  # (N, 2)

            # 负值截断为 0 后求面积
            intersect_area = torch.prod(torch.clamp_min(intersect_dims, 0), dim=1)  # (N,)

            # 累加该帧所有相交面积
            inter_sum += intersect_area.sum()

    return inter_sum * self.weight


def seg_utils_IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss, numerator, denominator


def clipmatcher_loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple]):
    """计算 bbox 相关 loss（L1 回归 + GIoU）。

    targets dicts 必须包含 "boxes" 键，值为 [nb_target_boxes, 4] 张量；
    目标框格式为 (center_x, center_y, h, w)，按图像尺寸归一化。
    """
    # 忽略消失 track 槽位的回归 loss
    filtered_idx = []
    for src_per_img, tgt_per_img in indices:
        keep = tgt_per_img != -1
        if not keep.all().item():
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
            indices = filtered_idx
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    sdc_boxes = outputs["pred_sdc_boxes"][0, -1:]
    target_sdc_boxes = gt_instances[0].sdc_boxes[:1]

    target_boxes_list = []
    target_obj_ids_list = []
    for gt_per_img, (_, i) in zip(gt_instances, indices):
        target_boxes_list.append(gt_per_img.boxes[i])
        target_obj_ids_list.append(gt_per_img.obj_ids[i])

    target_boxes = torch.cat(target_boxes_list, dim=0)
    target_obj_ids = torch.cat(target_obj_ids_list, dim=0)

    src_boxes = torch.cat([src_boxes, sdc_boxes], dim=0)
    target_boxes = torch.cat([target_boxes, target_sdc_boxes], dim=0)

    target_obj_ids = torch.cat([target_obj_ids, torch.zeros(1).to(target_obj_ids.device)], dim=0)
    mask = target_obj_ids != -1
    bbox_weights = torch.ones_like(target_boxes) * self.code_weights
    avg_factor = src_boxes[mask].size(0)
    avg_factor = reduce_mean(target_boxes.new_tensor([avg_factor]))
    loss_bbox = self.loss_bboxes(
        src_boxes[mask],
        target_boxes[mask],
        bbox_weights[mask],
        avg_factor=avg_factor.item(),
    )

    losses = {}
    losses["loss_bbox"] = loss_bbox

    return losses


def clipmatcher_match_for_single_frame(
    self,
    outputs: dict,
    dec_lvl: int,
    if_step=False,
):
    from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin import Instances
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

    outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

    gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
    track_instances: Instances = outputs_without_aux["track_instances"]
    pred_logits_i = track_instances.pred_logits
    pred_boxes_i = track_instances.pred_boxes
    # 硬编码的 sdc query 槽位
    pred_sdc_logits_i = track_instances.pred_logits[900:901].unsqueeze(0)
    pred_sdc_boxes_i = track_instances.pred_boxes[900:901].unsqueeze(0)
    track_instances.obj_idxes[900] = -2
    pred_past_trajs_i = track_instances.pred_past_trajs  # predicted past trajs of i-th image.

    obj_idxes = gt_instances_i.obj_ids
    obj_idxes_npu = obj_idxes.clone()

    outputs_i = {
        "pred_logits": pred_logits_i.unsqueeze(0),
        "pred_sdc_logits": pred_sdc_logits_i,
        "pred_boxes": pred_boxes_i.unsqueeze(0),
        "pred_sdc_boxes": pred_sdc_boxes_i,
        "pred_past_trajs": pred_past_trajs_i.unsqueeze(0),
    }
    # step1. 继承并更新上一次的 track。
    obj_idxes = track_instances.obj_idxes
    mask_valid = obj_idxes >= 0

    current_max = obj_idxes.max().item() if mask_valid.any() else 0

    lookup_size = max(current_max + 1, 1)
    lookup_table = torch.full((lookup_size,), -1, dtype=torch.long, device=track_instances.obj_idxes.device)

    if obj_idxes_npu.any():
        valid_mask = obj_idxes_npu < lookup_size
        valid_obj_idxes = obj_idxes_npu[valid_mask]
        valid_gt_indices = torch.arange(len(obj_idxes_npu), device=obj_idxes_npu.device)[valid_mask]
        lookup_table[valid_obj_idxes] = valid_gt_indices

    matched_gt_idxes = torch.where(
        mask_valid,
        lookup_table[torch.clamp(obj_idxes, min=-1, max=lookup_size - 1)],
        torch.tensor(-1, dtype=torch.long, device=track_instances.obj_idxes.device),
    )

    num_disappear_track = (mask_valid & (matched_gt_idxes == -1)).sum().item()
    track_instances.matched_gt_idxes = matched_gt_idxes

    full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
    # 之前已被跟踪、按规则匹配的
    matched_track_idxes = track_instances.obj_idxes >= 0
    prev_matched_indices = torch.stack(
        [
            full_track_idxes[matched_track_idxes],
            track_instances.matched_gt_idxes[matched_track_idxes],
        ],
        dim=1,
    ).to(pred_logits_i.device)

    # step2. 选出未匹配的槽位（FP tracks 的 obj_idxes==-2 不会进这里）
    unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

    # step3. 选出未被跟踪的 gt 实例（新轨迹）
    tgt_indexes = track_instances.matched_gt_idxes
    tgt_indexes = tgt_indexes[tgt_indexes != -1]

    tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
    tgt_state[tgt_indexes] = 1
    # 新的 tgt 索引
    untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
    # [num_untracked]
    untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

    def match_for_single_decoder_layer(unmatched_outputs, matcher):
        bbox_preds, cls_preds = (
            unmatched_outputs["pred_boxes"],
            unmatched_outputs["pred_logits"],
        )
        bs, num_querys = bbox_preds.shape[:2]
        # 拼接目标 label 与 box
        targets = [untracked_gt_instances]
        gt_labels_list = []
        gt_bboxes_list = []
        if isinstance(targets[0], Instances):
            for gt_per_img in targets:
                gt_labels_list.append(gt_per_img.labels)
                gt_bboxes_list.append(gt_per_img.boxes)
            gt_labels = torch.cat(gt_labels_list)
            gt_bboxes = torch.cat(gt_bboxes_list)
        else:
            for v in targets:
                gt_labels_list.append(v["labels"])
                gt_bboxes_list.append(v["boxes"])
            gt_labels = torch.cat(gt_labels_list)
            gt_bboxes = torch.cat(gt_bboxes_list)

        bbox_pred = bbox_preds[0]
        cls_pred = cls_preds[0]

        src_idx, tgt_idx = matcher.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
        if src_idx is None:
            return None
        # concat src and tgt.
        new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1).to(
            pred_logits_i.device
        )
        return new_matched_indices

    # step4. 对未匹配槽位与 GT 做匹配
    unmatched_outputs = {
        # [bs, num_pred, num_classes]
        "pred_logits": track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
        # [bs, num_pred, box_dim]
        "pred_boxes": track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
    }
    # [num_new_matched, 2]
    new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

    # step5. 按新匹配结果更新 obj_idxes
    if new_matched_indices is not None:
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. 计算 iou3d。
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        with torch.no_grad():
            if len(active_track_boxes) > 0:
                gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
                iou_3ds = iou_3d(
                    denormalize_bbox(gt_boxes, None)[..., :7],
                    denormalize_bbox(active_track_boxes, None)[..., :7],
                )
                track_instances.iou[active_idxes] = torch.tensor([iou_3ds[i, i] for i in range(gt_boxes.shape[0])]).to(
                    gt_boxes.device
                )

        # step7. 合并新匹配与既有匹配
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
    else:
        matched_indices = prev_matched_indices
    # step8. 计算 loss。
    self.num_samples += len(gt_instances_i) + num_disappear_track
    self.sample_device = pred_logits_i.device

    for loss in self.losses:
        new_track_loss = self.get_loss(
            loss,
            outputs=outputs_i,
            gt_instances=[gt_instances_i],
            indices=[(matched_indices[:, 0], matched_indices[:, 1])],
        )
        self.losses_dict.update(
            {
                "frame_{}_{}_{}".format(self._current_frame_idx, key, dec_lvl): value
                for key, value in new_track_loss.items()
            }
        )
    if "aux_outputs" in outputs:
        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            unmatched_outputs_layer = {
                "pred_logits": aux_outputs["pred_logits"][0, unmatched_track_idxes].unsqueeze(0),
                "pred_boxes": aux_outputs["pred_boxes"][0, unmatched_track_idxes].unsqueeze(0),
            }
            new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
            matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
            for loss in self.losses:
                if loss == "masks":
                    # 中间层 mask loss 代价过高，忽略
                    continue
                l_dict = self.get_loss(
                    loss,
                    aux_outputs,
                    gt_instances=[gt_instances_i],
                    indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                )
                self.losses_dict.update(
                    {
                        "frame_{}_aux{}_{}".format(self._current_frame_idx, i, key): value
                        for key, value in l_dict.items()
                    }
                )
    if if_step:
        self._step()
    return track_instances, matched_indices


def clipmatcher_prediction_loss(self, track_instances, predictions):
    decay_ratio = 1.0
    device = track_instances.obj_idxes.device
    for i in range(self._current_frame_idx, len(self.gt_instances)):
        gt_instances_i = self.gt_instances[i]  # gt instances of i-th image.

        pred_boxes_i = predictions[i - self._current_frame_idx]

        obj_idxes = gt_instances_i.obj_ids  # (num_gt,) on NPU

        # 矢量化匹配：在 NPU 上构建查找表
        track_obj_ids = track_instances.obj_idxes  # (num_tracks,)
        matched_gt = torch.full_like(track_obj_ids, -1)  # 默认 -1（未匹配/消失）

        # obj_id >= 0 的 track 才是有效（之前跟踪过）
        valid_tracks = track_obj_ids >= 0

        # 对每个有效 track，检查其 obj_id 是否在当前 gt 中
        # 广播：(num_tracks, 1) == (1, num_gt) -> (num_tracks, num_gt)
        # 然后 argmax 找到匹配的 gt 索引
        if obj_idxes.numel() > 0 and valid_tracks.any():
            eq_matrix = track_obj_ids.unsqueeze(1) == obj_idxes.unsqueeze(0)
            has_match = eq_matrix.any(dim=1)  # (num_tracks,)
            matched_gt_idx = eq_matrix.int().argmax(dim=1)  # argmax 取第一个匹配
            matched_gt = torch.where(has_match & valid_tracks, matched_gt_idx, torch.tensor(-1, device=device))

        track_instances.matched_gt_idxes = matched_gt
        num_paired = (matched_gt >= 0).sum().item()  # 用于控制流的标量，比 dict 循环便宜

        if num_paired > 0:
            if_paired_i = matched_gt >= 0

            paired_pred_boxes_i = pred_boxes_i[if_paired_i]

            paired_gt_instances = gt_instances_i[matched_gt[if_paired_i]]
            normalized_bboxes = paired_gt_instances.boxes
            cx = normalized_bboxes[..., 0:1]
            cy = normalized_bboxes[..., 1:2]
            cz = normalized_bboxes[..., 4:5]

            gt_boxes_i = torch.cat([cx, cy, cz], dim=-1)

            pred_loss_i = 0.2 * decay_ratio * self.loss_predictions(paired_pred_boxes_i, gt_boxes_i).sum(dim=-1).mean()

            self.losses_dict["pred_loss_{}".format(i)] = pred_loss_i
        else:
            self.losses_dict["pred_loss_{}".format(i)] = torch.tensor([0.0]).to(device)

        decay_ratio = decay_ratio * 0.5


# ---------------------------------------------------------------------------
# 4) NPU 侧匈牙利匹配
#
# core/bbox/assigners/hungarian_assigner_3d_track.py 与
# seg_head_plugin/seg_assigner.py（以及配置层 HungarianAssigner→Detr 的切换）：
# 把 scipy/CPU 的 ``linear_sum_assignment`` 换成 NPU 侧
# ``mx_driving.hungarian_match`` 算子（无 H2D/D2H 拷贝）。
# 注意：本环境使用 ``mx_driving.hungarian_match``（返回 (row, col) 两个
# int32 张量）；uniad.patch 源码里同一算子名为 ``mx_driving.hungarian``。
# ---------------------------------------------------------------------------

INF = 10000000


def assign_mmdet_hungarian(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore=None, eps=1e-7):
    """mmdet HungarianAssigner.assign 的 NPU 版（DETR 式 cost 匹配）。"""
    from mx_driving import hungarian_match

    assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    # 1. 默认全部赋 -1
    assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
    assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
    if num_gts == 0 or num_bboxes == 0:
        # 无 gt 或无框时返回空分配
        if num_gts == 0:
            # 无 gt，全部归为背景
            assigned_gt_inds[:] = 0
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    # 2. 计算加权 cost（与 mmdet 原实现一致：img_shape 归一化 + cxcywh→xyxy）
    img_h, img_w, _ = img_meta['img_shape']
    factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    cls_cost = self.cls_cost(cls_pred, gt_labels)
    normalize_gt_bboxes = gt_bboxes / factor
    reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
    bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
    iou_cost = self.iou_cost(bboxes, gt_bboxes)
    cost = cls_cost + reg_cost + iou_cost

    # 3. NPU 侧匈牙利匹配（无 h2d/d2h；返回 int32，转 long 供下游使用）
    matched_row_inds, matched_col_inds = hungarian_match(cost)
    matched_row_inds = matched_row_inds.long()
    matched_col_inds = matched_col_inds.long()

    # 4. 先全部归背景，再按匹配结果赋前景
    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


def assign_hungarian_3d_track(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-7):
    from mx_driving import hungarian_match

    """HungarianAssigner3DTrack.assign 的 NPU 版。

        计算 cls + reg 加权 cost 后在 NPU 侧做匈牙利匹配，返回
        (matched_row_inds, matched_col_inds)（track_head 使用）。
        """
    assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    # 1. 默认全部赋 -1
    assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
    assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
    if num_gts == 0 or num_bboxes == 0:
        # 无 gt 或无框时返回空分配
        if num_gts == 0:
            # 无 gt，全部归为背景
            assigned_gt_inds[:] = 0
        return (None, None)
    # 2. 计算加权 cost
    cls_cost = self.cls_cost(cls_pred, gt_labels)
    reg_cost = self.reg_cost(bbox_pred[:, :8], gt_bboxes[:, :8])
    cost = cls_cost + reg_cost

    # 3. NPU 侧匈牙利匹配（无 h2d/d2h；返回 int32，转 long 供下游使用）
    matched_row_inds, matched_col_inds = hungarian_match(cost)
    matched_row_inds = matched_row_inds.long()
    matched_col_inds = matched_col_inds.long()

    # 4. 先全部归背景，再按匹配结果赋前景
    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

    return (matched_row_inds, matched_col_inds)


def assign_seg_filter(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore=None, eps=1e-7):
    from mx_driving import hungarian_match

    """HungarianAssigner_filter.assign 的 NPU 版（panseg 的 filter 分支）。

        与原实现一致：cost 每轮取最优匹配后把已匹配行置 INF，循环最多
        min(max_pos, 300//num_gts) 轮，返回 (pos_ind, neg_ind, AssignResult)。
        """
    assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    # 1. 默认全部赋 -1
    assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

    assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

    if num_gts == 0 or num_bboxes == 0:
        # 无 gt 或无框时返回空分配
        if num_gts == 0:
            # 无 gt，全部归为背景
            assigned_gt_inds[:] = 0
        pos_ind = assigned_gt_inds.gt(0).nonzero().squeeze(1)
        neg_ind = assigned_gt_inds.eq(0).nonzero().squeeze(1)
        return pos_ind, neg_ind, AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    img_h, img_w, _ = img_meta['img_shape']
    factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

    # 2. 计算加权 cost（cls + reg + iou）
    cls_cost = self.cls_cost(cls_pred, gt_labels)
    normalize_gt_bboxes = gt_bboxes / factor
    reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
    bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
    iou_cost = self.iou_cost(bboxes, gt_bboxes)
    cost = cls_cost + reg_cost + iou_cost

    # 3. NPU 侧匈牙利匹配：多轮选择（与原实现 linear_sum_assignment 一致）
    assigned_gt_inds[:] = 0

    result = None
    for i in range(min(self.max_pos, 300 // num_gts)):
        matched_row_inds, matched_col_inds = hungarian_match(cost)
        matched_row_inds = matched_row_inds.long()
        matched_col_inds = matched_col_inds.long()
        cost[matched_row_inds, :] = INF
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        if i == 0:
            result = AssignResult(num_gts, assigned_gt_inds.clone(), None, labels=assigned_labels.clone())

        if cost[matched_row_inds, matched_col_inds].max() >= INF:
            break
    pos_ind = assigned_gt_inds.gt(0).nonzero().squeeze(1)
    neg_ind = assigned_gt_inds.eq(0).nonzero().squeeze(1)

    return pos_ind, neg_ind, result


def assign_seg_multi(
    self, bbox_pred, cls_pred, mask_pred, gt_bboxes, gt_labels, gt_mask, img_meta, gt_bboxes_ignore=None, eps=1e-7
):
    from mx_driving import hungarian_match

    """HungarianAssigner_multi_info.assign 的 NPU 版（panseg with-mask 分支）。

        在 cls+reg+iou 之上叠加 mask cost 后做 NPU 侧匈牙利匹配。
        """
    assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    # 1. 默认全部赋 -1
    assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

    assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

    if num_gts == 0 or num_bboxes == 0:
        # 无 gt 或无框时返回空分配
        if num_gts == 0:
            # 无 gt，全部归为背景
            assigned_gt_inds[:] = 0
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    img_h, img_w, _ = img_meta['img_shape']

    factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

    # 分类与 bbox cost
    cls_cost = self.cls_cost(cls_pred, gt_labels)
    # 回归 L1 cost
    normalize_gt_bboxes = gt_bboxes / factor
    reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
    # 回归 iou cost（官方 DETR 默认 giou）
    bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
    iou_cost = self.iou_cost(bboxes, gt_bboxes)
    # mask cost
    mask_cost = self.mask_cost(mask_pred, gt_mask)
    #
    cost = cls_cost + reg_cost + iou_cost + mask_cost

    # 3. NPU 侧匈牙利匹配（无 h2d/d2h；返回 int32，转 long 供下游使用）
    matched_row_inds, matched_col_inds = hungarian_match(cost)
    matched_row_inds = matched_row_inds.long()
    matched_col_inds = matched_col_inds.long()
    # 4. 先全部归背景，再按匹配结果赋前景
    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


# =============================================================================
# 二阶段补丁类（全部使用 AtomicPatch）
# =============================================================================


class UniADSmootherPool(Patch):
    """并行（pool）+ 异步 smoother 入口（对应 motion_utils.py hunk）。"""

    name = "smoother_pool"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        base = "projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.motion_utils"
        return [
            AtomicPatch(f"{base}.nonlinear_smoother", nonlinear_smoother),
            AtomicPatch(f"{base}.submit_nonlinear_smoother", submit_nonlinear_smoother),
            AtomicPatch(f"{base}.SmootherAsyncResult", SmootherAsyncResult),
        ]


class UniADSmootherMotionHead(Patch):
    """重新绑定 motion_head 模块里的 ``nonlinear_smoother``（按名 import 处）。"""

    name = "smoother_motion_head"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.dense_heads.motion_head.nonlinear_smoother",
                nonlinear_smoother,
            )
        ]


class UniADOccHeadVectorize(Patch):
    """矢量化 OccHead.forward_train 的逐 query GT 目标循环（对应 occ_head.py hunk）。"""

    name = "occ_head_vectorize"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.dense_heads.occ_head.OccHead.forward_train",
                forward_train,
            )
        ]


class UniADPlanningLossVectorize(Patch):
    """矢量化 CollisionLoss.forward（规划 loss 全部在 device 侧计算）。"""

    name = "planning_loss_vectorize"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.losses.planning_loss.CollisionLoss.forward",
                collision_loss_forward_vectorized,
            )
        ]


class UniADSegUtilsNoD2H(Patch):
    """去掉 seg_utils.IOU 末尾的 .cpu()（D2H）调用。"""

    name = "seg_utils_no_d2h"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.uniad.dense_heads.seg_head_plugin.seg_utils.IOU",
                seg_utils_IOU,
            )
        ]


class UniADTrackLossVectorize(Patch):
    """矢量化 ClipMatcher 的 NPU 热路径（loss_boxes / 匹配 / prediction_loss）。"""

    name = "track_loss_vectorize"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        base = "projects.mmdet3d_plugin.losses.track_loss.ClipMatcher"
        return [
            AtomicPatch(f"{base}.loss_boxes", clipmatcher_loss_boxes),
            AtomicPatch(f"{base}.match_for_single_frame", clipmatcher_match_for_single_frame),
            AtomicPatch(f"{base}.prediction_loss", clipmatcher_prediction_loss),
        ]


class UniADHungarianAssignerNPU(Patch):
    """mmdet HungarianAssigner.assign → NPU 侧匹配（对应配置层 assigner 切换）。"""

    name = "hungarian_assigner_npu"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "mmdet.core.bbox.assigners.HungarianAssigner.assign",
                assign_mmdet_hungarian,
            )
        ]


class UniADHungarianAssigner3DTrackNPU(Patch):
    """HungarianAssigner3DTrack.assign → NPU 侧匹配（track_head 常规路径）。"""

    name = "hungarian_assigner_3d_track_npu"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d_track.HungarianAssigner3DTrack.assign",
                assign_hungarian_3d_track,
            )
        ]


class UniADSegAssignerNPU(Patch):
    """seg_assigner 的两个 assigner（filter / multi_info）→ NPU 侧匹配。"""

    name = "seg_assigner_npu"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        base = "projects.mmdet3d_plugin.uniad.dense_heads.seg_head_plugin.seg_assigner"
        return [
            AtomicPatch(f"{base}.HungarianAssigner_filter.assign", assign_seg_filter),
            AtomicPatch(f"{base}.HungarianAssigner_multi_info.assign", assign_seg_multi),
        ]


GENERIC_EXTRA_PATCHES: tuple = (
    CollectionsCompat,
    NpuFusedAdamwStep,
    SigmoidFocalLoss,
)

UNIAD_RUNTIME_PATCHES = (
    UniADBootstrapCompat,
    UniADConfigPatch,
    UniADLocalMSDA,
    UniADDataPipeline,
    UniADBBox,
    UniADDataBuilder,
    UniADTrajLoss,
    UniADMapApi,
    UniADFunctional,
    UniADTraining,
    UniADMotionHead,
    UniADPlanningMetric,
    UniADDetector,
    UniADEncoder,
    UniADSpatialCrossAttention,
    UniADTransformer,
)

UNIAD_RUNTIME_MODULE_PATCHES = (
    UniADSmootherPool,
    UniADSmootherMotionHead,
    UniADOccHeadVectorize,
    UniADPlanningLossVectorize,
    UniADSegUtilsNoD2H,
    UniADTrackLossVectorize,
    UniADHungarianAssignerNPU,
    UniADHungarianAssigner3DTrackNPU,
    UniADSegAssignerNPU,
)

_UNIAD_MODULE_PATCH_NAMES = {cls.name for cls in UNIAD_RUNTIME_MODULE_PATCHES}


def _disabled_uniad_patches_from_env() -> set:
    """解析 ``UNIAD_PATCH_DISABLE``（逗号分隔的名字或 ``*``）。"""
    raw = os.environ.get("UNIAD_PATCH_DISABLE", "")
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if "*" in parts:
        return set(_UNIAD_MODULE_PATCH_NAMES)
    return set(parts) & _UNIAD_MODULE_PATCH_NAMES


def configure_patcher(patcher: Patcher, performance: bool = False) -> Patcher:
    """配置 mx_driving patcher 并返回，供 migrate_to_ascend/train.py 调用。

    Args:
        patcher: ``mx_driving.patcher.default_patcher``（或新建的 ``Patcher()``）。
        performance: 为 True 时 ``brake_at(UNIAD_BRAKE_AT)``（默认 500）提前
            停止训练循环，用于 smoke/perf 运行。
    """
    _set_hf32_flags()

    patcher.replace_import("nuscenes.eval.tracking.mot", "migrate_to_ascend.nuscenes_mot")
    patcher.disallow_internal_format()
    patcher.disable(TensorIndex, BatchMatmul)

    for name in _disabled_uniad_patches_from_env():
        patcher.disable(name)

    patcher.add(
        AtomicPatch("torch.bmm", _matmul_contiguous),
        AtomicPatch("torch.matmul", _matmul_contiguous),
    )

    patcher.add(*GENERIC_EXTRA_PATCHES)
    patcher.add(*UNIAD_RUNTIME_PATCHES)
    patcher.add(*UNIAD_RUNTIME_MODULE_PATCHES)

    if performance:
        patcher.brake_at(int(os.environ.get("UNIAD_BRAKE_AT", "500")))

    return patcher
