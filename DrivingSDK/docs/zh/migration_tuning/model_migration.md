# 模型迁移

模型迁移指将基于CUDA生态开发的开源自动驾驶模型适配到昇腾NPU平台，使其能够在昇腾上完成训练与推理。Driving SDK提供两条迁移路径：

- **传统迁移（手动代码替换）**：直接修改模型源码，逐个替换CUDA算子、适配分布式训练、调整三方库依赖。
- **进阶能力（一键Patcher自动替换）**：基于Monkey Patch的运行时替换，不修改原始代码，几行代码即可完成适配。

## 迁移方式概览

两条迁移路径的实现原理与适用场景差异如下：

| 对比维度 | 传统迁移（手动代码替换） | 一键Patcher |
|----------|--------------------------|--------------|
| 实现原理 | 直接修改模型源码 | 运行时Monkey Patch，零侵入 |
| 改动范围 | 修改点散落在模型源码多个文件 | 迁移代码集中于独立目录，原始代码不动 |
| 三方库依赖 | 修改mmcv等三方库后需重新编译 | 运行时替换，无需重新编译 |
| 升级与维护 | 上游更新后需重新适配 | 补丁独立于模型代码，可复用 |
| 适用场景 | 需深度改造模型结构、算子无现成补丁 | 标准CUDA算子替换与常见依赖处理 |
| 实施门槛 | 高，需逐点排查修改 | 低，配置即用 |

本文第一部分（传统迁移）以BEVDet模型为例演示手动迁移的完整流程，第二部分（进阶能力）介绍一键Patcher的快速用法。

## 传统迁移：手动代码替换

传统迁移直接在模型源码上进行修改，典型操作包括：

- 将CUDA专属算子替换为NPU实现（或Driving SDK融合算子）；
- 将分布式训练接口（DDP）适配为NPU版本；
- 调整三方库版本兼容性、去除CUDA相关代码；
- 适配NPU的算子约束（如matmul维度限制）。

以下以BEVDet模型为例，演示完整的传统迁移流程。

### 为何选用BEVDet作为示例

BEVDet是基于俯视图的3D目标检测模型，在自动驾驶、智能交通等领域应用广泛。其迁移过程覆盖了传统迁移的典型环节，便于读者举一反三：

| 迁移环节 | BEVDet中的体现 |
|----------|-----------------|
| 算子替换 | `bev_pool_v2`（CUDA）替换为Driving SDK的`bev_pool_v3` |
| 分布式适配 | `MMDataParallel` / `MMDistributedDataParallel`适配为NPU版本 |
| 三方库兼容 | mmcv版本兼容适配、去除CUDA相关代码 |
| NPU约束适配 | matmul不支持6维以上张量的适配 |

迁移完成、模型可在NPU上正常运行后，可参考[模型优化](model_optimization.md)对模型做进一步的性能优化。

### 环境与依赖准备

1. **参考实现**

    - 原始实现：`https://github.com/HuangJunJie2017/BEVDet.git`，commit_id=`58c2587a8f89a1927926f0bdb6cde2917c91a9a5`
    - 适配昇腾AI处理器的实现：`https://gitcode.com/Ascend/DrivingSDK.git`，code_path=`model_examples/BEVDet`

2. **安装模型依赖**

    自驾模型当前主要以PyTorch 2.1进行配置，以下以PyTorch 2.1为例。创建`requirements.txt`，添加如下依赖及版本：

    ```python
    setuptools==65.7.0
    torchvision==0.16.0
    nuscenes-devkit==1.1.11
    numba==0.58.1
    numpy==1.23.1
    lyft_dataset_sdk
    scikit-image
    trimesh==2.35.39
    tensorboard
    networkx
    attrs
    decorator
    sympy
    cffi
    pyyaml
    pathlib2
    psutil
    protobuf==4.25.0
    scipy
    requests
    absl-py
    yapf
    mmdet==2.28.2
    mmsegmentation==0.30.0
    ninja
    ```

    通过`pip install -r requirements.txt`安装。

3. **编译安装mmcv**

    mmcv 1.7.2版本支持NPU，需要手动下载代码进行源码编译安装：

    ```python
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    ```

    适配PyTorch 2.x版本，修改`mmcv/mmcv/parallel/distributed.py`第159行，原始代码为：

    ```python
    module_to_run = self._replicated_tensor_module if self._use_replicated_tensor_module else self.module
    ```

    替换为：

    ```python
    module_to_run = self.module
    ```

    进入mmcv根目录，使用如下指令编译安装：

    ```python
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
    ```

### 迁移适配

> [!NOTE]
>
> 本节步骤仅保证模型在昇腾环境可运行，不涉及性能优化。迁移完成后如需提升性能，参考[模型优化](model_optimization.md)。

1. **下载模型官方源码并指定commit id**

    ```python
    git clone https://github.com/HuangJunJie2017/BEVDet.git
    cd BEVDet
    git checkout 58c2587a8f89a1927926f0bdb6cde2917c91a9a5
    ```

2. **在`tools/train.py`和`tools/test.py`中添加自动迁移代码**

    ```python
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    ```

3. **DDP调用适配为NPU版本**

    修改`tools/test.py`，将原始代码：

    ```python
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    ```

    改为：

    ```python
    from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
    ```

    并将原始代码：

    ```python
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    …
    model = MMDistributedDataParallel(
    model.cuda(),
    ```

    改为：

    ```python
    model = NPUDataParallel(model.npu(), device_ids=cfg.gpu_ids)
    …
    model = NPUDistributedDataParallel(
    model.npu(),
    ```

    修改`mmdet3d/apis/train.py`，将原始代码：

    ```python
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    ```

    改为：

    ```python
    from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
    ```

    将代码中相应的代码：

    ```python
    model = MMDistributedDataParallel(
    model = MMDataParallel(
    ```

    改为：

    ```python
    model = NPUDistributedDataParallel(
    model = NPUDataParallel(
    ```

4. **适配PyTorch 2.1，添加`--local-rank`参数**

    在`tools/train.py`和`tools/test.py`中原始代码：

    ```python
    parser.add_argument('--local_rank', type=int, default=0)
    ```

    前添加：

    ```python
    parser.add_argument('--local-rank', type=int, default=0)
    ```

5. **去除CUDA相关代码**

    修改`mmdet3d/models/detectors/__init__.py`，将原始代码：

    ```python
    from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVDetTRT, BEVStereo4D
    ```

    改为：

    ```python
    from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVStereo4D
    ```

    注释掉原始代码：

    ```python
    from .dal import DAL
    ```

    将原始代码：

    ```python
    all = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
    'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
    'BEVDetTRT', 'BEVStereo4D', 'BEVStereo4DOCC'
    ]
    ```

    改为：

    ```python
    all = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
    'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
    'BEVStereo4D', 'BEVStereo4DOCC'
    ]
    ```

    修改`mmdet3d/models/detectors/bevdet.py`，删除或注释如下代码：

    ```python
    from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
    …
    @DETECTORS.register_module()
    class BEVDetTRT(BEVDet):

        def result_serialize(self, outs):
            outs_ = []
            for out in outs:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
            return outs_

        def result_deserialize(self, outs):
            outs_ = []
            keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
            for head_id in range(len(outs) // 6):
                outs_head = [dict()]
                for kid, key in enumerate(keys):
                    outs_head[0][key] = outs[head_id * 6 + kid]
                outs_.append(outs_head)
            return outs_

        def forward(
            self,
            img,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
        ):
            x = self.img_backbone(img)
            x = self.img_neck(x)
            x = self.img_view_transformer.depth_net(x)
            depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
            tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
            tran_feat = tran_feat.permute(0, 2, 3, 1)
            x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                                ranks_depth, ranks_feat, ranks_bev,
                                interval_starts, interval_lengths)
            x = x.permute(0, 3, 1, 2).contiguous()
            bev_feat = self.bev_encoder(x)
            outs = self.pts_bbox_head([bev_feat])
            outs = self.result_serialize(outs)
            return outs

        def get_bev_pool_input(self, input):
            input = self.prepare_inputs(input)
            coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
            return self.img_view_transformer.voxel_pooling_prepare_v2(coor)
    ```

6. **mmdet_3d版本兼容适配**

    修改`mmdet3d/__init__.py`，将原始代码：

    ```python
    mmcv_maximum_version = '1.7.0'
    ```

    改为：

    ```python
    mmcv_maximum_version = '1.7.2'
    ```

    注释或删除如下原始代码：

    ```python
    import mmseg
    …
    mmseg_version = digit_version(mmseg.__version__)
    assert (mmseg_version >= digit_version(mmseg_minimum_version)
        and mmseg_version <= digit_version(mmseg_maximum_version)),
        f'MMSEG=={mmseg.__version__} is used but incompatible. '
        f'Please install mmseg>={mmseg_minimum_version}, '
        f'<={mmseg_maximum_version}.'
    ```

7. **替换BEV Pooling算子为Driving SDK版本**

    原始代码中`bev_pool_v2`算子为CUDA代码，需要替换为Driving SDK仓的`bev_pool_v3`（NPU亲和实现，详见[模型优化](./model_optimization.md)文档的[替换融合算子](./model_optimization.md#替换融合算子)章节）。安装Driving SDK参考<https://gitcode.com/Ascend/DrivingSDK/blob/master/README.md>。

    修改`mmdet3d/models/necks/view_transformer.py`，删除原始代码：

    ```python
    from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
    ```

    并加入引用：

    ```python
    from mx_driving.point import bev_pool_v3
    ```

    将`LSSViewTransformer`类中`voxel_pooling_v2`函数的原始代码：

    ```python
    bev_feat = bev_pool_v2(depth, feat, ranks_depth,
            ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts,
            interval_lengths)
    ```

    替换为：

    ```python
    bev_feat = bev_pool_v3(depth, feat, ranks_depth,
            ranks_feat, ranks_bev,
            bev_feat_shape)
    ```

    将`LSSViewTransformer`类中`view_transform_core`函数的原始代码：

    ```python
    bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
            self.ranks_feat, self.ranks_bev,
            bev_feat_shape, self.interval_starts,
            self.interval_lengths)
    ```

    替换为：

    ```python
    bev_feat = bev_pool_v3(depth, feat, self.ranks_depth,
            self.ranks_feat, self.ranks_bev,
            bev_feat_shape)
    ```

8. **适配matmul维度限制**

    NPU上matmul不支持6维以上张量，需要修改适配。替换`mmdet3d/models/necks/view_transformer.py`中`LSSViewTransformer`类的`get_lidar_coor`函数：

    ```python
    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        B, N, D, H, W, _ = points.shape
        points = points.view(B, N, D*H*W, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 3, 3).matmul(points)

        # cam_to_ego
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 3)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 3)
        points = bda[:, :3, :3].view(B, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 3)
        return points.view(B, N, D, H, W, 3)
    ```

完成修改后，BEVDet模型即可在NPU上进行训练。本文档仅给出模型适配迁移的具体步骤，若需更详细的模型训练指导，请参见[BEVDet模型文档](../../../model_examples/BEVDet/README.md)。

## 进阶能力：一键Patcher

Driving SDK提供了一键Patcher能力，可实现高效便捷的模型迁移操作，帮助用户仅需添加几行代码即可使模型在昇腾NPU上进行训练。

> [!NOTE]
>
> 以下步骤仅指导开发者如何通过一键Patcher快速进行模型迁移。一键Patcher的快速上手指南见[一键Patcher（快速上手）](../features/patcher.md)，各功能的底层逻辑、实现原理与接口说明见[一键Patcher（功能详解）](../features/patcher_features.md)。

### 最简用法

在训练脚本（通常是`train.py`）的**最顶部**（所有其他import之前）添加：

```python
# train.py最顶部
from mx_driving.patcher import default_patcher
default_patcher.apply()

# ↓↓↓ 以下是模型原始代码示例，以实际使用模型为准，一个字符都不需要改 ↓↓↓
import mmcv
import torch
from my_model import build_model
...
```

> [!NOTE]
>
> **为什么要放在最上方？** Patcher通过Monkey Patch机制替换目标模块的函数/类。如果目标模块在`apply()`之前已被导入，补丁可能无法正确生效。
>
> **关于torch/torch_npu依赖：** Patcher内部会自动处理torch和torch_npu的导入，用户无需在导入Patcher之前手动导入这些模块。

添加完成后，正常运行模型即可。若模型还有未覆盖的自定义CUDA算子，可定义自定义补丁并加入`default_patcher`，写法参考[一键Patcher（快速上手）：在default_patcher之上扩展](../features/patcher.md#在default_patcher之上扩展)。

## 两种方式如何选择

| 判断条件 | 推荐方式 |
|----------|----------|
| 迁移项为标准CUDA算子替换、缺失依赖处理、导出缺失 | 一键Patcher |
| 需要深度改造模型结构（如替换整个网络模块） | 传统迁移（手动代码替换） |
| 算子没有现成NPU补丁、需要精细控制实现 | 传统迁移（手动代码替换） |
| 已有模型通过手动方式完成过类似迁移，改动可复用 | 传统迁移（手动代码替换） |

建议的实践路径：

- **先用一键Patcher快速跑通**：多数模型的迁移项集中在算子替换与依赖处理，Patcher成本低、可复用，适合作为首选。
- **再针对个别算子手动替换**：对Patcher未覆盖或需深度定制的模块，用传统手动方式补充，两种方式可结合使用。
- **传统迁移更适合作为学习与排障基础**：理解手动替换的各个环节，有助于在Patcher不生效时快速定位问题。
