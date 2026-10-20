# Model Optimization

## Model Optimization Guide

### Framework Features

The TorchNPU framework enables a variety of features, including environment variable configuration, framework compilation optimization, and high-performance memory libraries.

#### Common Environment Variable Configuration and Introduction

| Environment Variable Configuration | Meaning |
|--------------|-------|
| export TASK_QUEUE_ENABLE=2 | Enables or disables task queue. `0` indicates disabled, `1` indicates level 1 optimization, `2` indicates level 2 optimization. |
| export CPU_AFFINITY_CONF=1 | Configures task CPU affinity to reduce scheduling overhead. `0` indicates disabled, `1` indicates coarse-grained CPU affinity, `2` indicates fine-grained CPU affinity. |
| export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" | Enables expandable segments for the memory pool. This setting instructs the caching allocator to create expandable memory segments. |
| export COMBINED_ENABLE=1 | Optimizes scenarios that combine two non-contiguous operators. `0` indicates disabled, `1` indicates enabled. |

#### Compilation Optimization

The framework supports compilation optimization for Python, PyTorch, and TorchNPU. This can reduce link time and memory usage. For details, see the following:

| Compilation Optimization Method | Reference Link |
|--------------|-------|
| Python | [Python Compilation Optimization](https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/comp_opt_py.md) |
| PyTorch | [PyTorch Compilation Optimization](https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/pytorch_comp_opt.md) |
| TorchNPU | [TorchNPU Compilation Optimization](https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/torch_npu_comp_opt.md) |

#### Replacing the High-Performance Memory Library

`tcmalloc`, or Thread-Caching Malloc, is a general-purpose memory allocator. By introducing a multilevel caching structure, reducing mutex contention, and optimizing large-object handling, it improves overall performance while maintaining low latency. This is especially important for applications that frequently perform memory operations, and it can significantly improve system responsiveness and service quality in highly concurrent scenarios. For details, see [Replacing the High-Performance Memory Library](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/performance_tuning/performance_tuning_methods/scheduling_opt.md#high-performance-memory-library-replacement).

### Replacing High-Performance Operators

Using fused operators can reduce small operator launches and improve model performance.
First, replace the fused optimizer. For details, see [Fused Optimizer](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/performance_tuning/performance_tuning_methods/npu_affinity_opt/fusion_optimizer.md).

Autonomous driving models usually use the AdamW optimizer. Replace it as follows:

| Optimizer | Sample Source Code | Updated Code |
|--------------|-------|----------|
| AdamW | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) | optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=args.lr) |

You can also replace compute-fused operators. For an introduction to the high-performance operators in the Driving SDK repository, see [Operator List](../api/README.md).

Common fused operators include the following:

| Operator | Usage |
| ------| ---------|
| deform_conv2d | `from mx_driving import deform_conv2d, DeformConv2dFunction` |
| npu_add_relu | `from mx_driving import npu_add_relu`. This is suitable for optimizing models that use ResNet as the backbone. It replaces `out += identity` and `out = self.relu(out)`. |
| npu_max_pool2d | `from mx_driving import npu_max_pool2d`. This replaces the `nn.MaxPool2d` operation. |
| multi_scale_deformable_attn | `from mx_driving import multi_scale_deformable_attn` |
| bev_pool_v3 | `from mx_driving import bev_pool_v3` |
| SparseConv3d | `from mx_driving import SparseConv3d` |
| SubMConv3d | `from mx_driving import SubMConv3d` |

#### Operator Replacement Example

After collecting profiling data, check the operator time statistics, analyze the operators that account for a large time share, and replace them. This section uses the BEVFormer model as an example to guide you through operator replacement.

The operator time statistics show that the forward and backward passes of `multi_scale_deformable_attn` account for a large share of the model. Therefore, you need to replace it with the NPU-friendly operator from `mx_driving` to improve performance.

![Operator time share](./figures/op_statistic.png)

> [!NOTE]
>
> - The operator is used in `projects/mmdet3d_plugin/bevformer/modules/decoder.py`.
> - For detailed operator parameters, see [Operator List](../api/README.md).

1. Import `mx_driving` in the header file.

    ```python
    import mx_driving
    ```

2. Replace the original operator with the NPU-friendly operator from `mx_driving`.

    Find the location where the `multi_scale_deformable_attn` operator is used (line 325). The original code is:

    ```python
    if torch.cuda.is_available() and value.is_cuda:

        # Using fp16 deformable attention is unstable because it performs many sum operations
        if value.dtype == torch.float16:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        else:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)
    else:
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
    ```

    Replace it with the operator from the `mx_driving` repository:

    ```python
        output = mx_driving.multi_scale_deformable_attn(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
    ```

### Optimizing Host Bound Issues

Autonomous driving algorithms contain many small operators such as slice, gather, and sort. Converting 2D image features to BEV space involves many operations such as projection, interpolation, and sampling, as well as voxelization of point cloud data and sparse convolutions. Autonomous driving algorithms also involve extensive logical processing, such as target assignment in detection algorithms, lovasz loss in planning and control algorithms, and operations that process ground truth.

Host-bound issues can become severe when many small operators are launched, when the CPU handles a large amount of logical processing, or when there is no high-performance NPU replacement for an operator.

#### Reducing the Number of Operator Launches

In lane detection, polygon instance segmentation, and similar tasks, targets usually consist of continuous point sequences. Through interpolation supervision, the model can learn this continuous geometric structure better. For example, in PivotNet, the `interpolate` function uses a loop to calculate one interpolation point at a time. This function is called 90 times in a single step, which introduces many small operators and idle time.

The main way to reduce operator launches is to replace scripts with equivalent APIs. Decision modules in planning and control models and data processing modules in perception models often contain many loop and slice operations because open-source model code usually lacks optimization. You can replace this large amount of small-operator logic with mask-related APIs.

In tasks such as object detection, targets usually consist of continuous point sequences, and interpolation-related logic often appears. If the model performs point-by-point calculations in a loop, try changing it to coarse-grained operations to reduce small operator launches.
For example, the model source code may contain loop-based point-by-point interpolation logic:

```python
@staticmethod
def interpolate(start_pt, end_pt, inter_num):
    res = torch.zeros((inter_num, 2), dtype=start_pt.dtype, device=start_pt.device)
    num_len = inter_num + 1 # Segment num
    for i in range(1, num_len):
        ratio = i / num_len
        res[i-1] = (1 - ratio) * start_pt + ratio * end_pt
    return res
```

You can vectorize the logic to increase the granularity of data processing and avoid loop-based point-by-point interpolation. The optimized code is:

```python
@staticmethod
def interpolate(start_pt, end_pt, inter_num):
    ratios = torch.arange(1, inter_num + 1, dtype=start_pt.dtype, device=start_pt.device) / (inter_num + 1)
    ratios = ratios.view(-1, 1)
    return (1 - ratios) * start_pt + ratios * end_pt
```

#### Converting CPU Computation into Small NPU Operator Composition

Planning and control models are small, and their end-to-end latency is relatively short. Therefore, if CPU computation still exists in the model, it becomes a serious performance bottleneck. The following figure shows the `torch.linalg.slogdet` interface used in the MultiPath++ model to evaluate the difference between predicted trajectories and ground truth trajectories. However, the current `torch.nn.linalg` linear algebra library does not have a corresponding NPU operator on Ascend. Therefore, native PyTorch CPU computation is used. This introduces a long CPU computation time, and you can replace it with an equivalent script that composes small operators.

The model source code is as follows:

```python
def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
```

You can replace it with:

```python
def custom_logdet(covariance_matrices):
    U, S, V = torch.svd(covariance_matrices)
    abs_det = torch.prod(S)
    return torch.log(abs_det)

def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * custom_logdet(covariance_matrices).unsqueeze(-1))
```

#### Reducing Pipeline Stalls Caused by Synchronization

The Hungarian matching algorithm is widely used in object detection, instance segmentation, and similar models to match predicted objects and ground truth objects. In PivotNet, the `forward` method of the Hungarian algorithm introduces many `aclrtSynchronizeStream` calls, which break the model pipeline. Common causes include `aten::item`, `.npu()`, and `.cpu()` calls.
For example, the following code generates a mask `gt_pts_mask` from `valid_lens` that represents the valid length of each sample. Positions within the valid length are 1, and the remaining positions are 0. Fetching `ll` implicitly calls `.item()`, which breaks the pipeline:

```python
for i, ll in enumerate(targets[0]["valid_len"][cid]):
    gt_pts_mask[i][:ll] = 1
```

You can avoid this by rewriting the script:

```python
valid_lens = torch.tensor(targets[0]["valid_len"][cid], device=gt_pts.device, dtype=torch.long)
row_indices = torch.arange(n_pt, device=gt_pts.device).expand(len(valid_lens), -1)
gt_pts_mask = (row_indices < valid_lens.unsqueeze(1)).to(dtype=torch.float32)
```

## Model Optimization Example

Take the BEVDet model as an example to demonstrate the complete model migration optimization process.

### Model Introduction

BEVDet is a deep learning model for 3D object detection. It can detect objects in 3D space from a bird's-eye-view image and predict their position, size, and orientation. It is widely used in autonomous driving, intelligent transportation, and related fields. Based on deep learning technology, it uses convolutional neural networks and residual networks. During training, it uses a large amount of 3D bounding box data to optimize model performance and accuracy.

The following optimization methods improve model performance:

| Optimization Method | File to Modify |
| -------  | --------|
| `export TASK_QUEUE_ENABLE=2` to enable L2 pipeline optimization | Environment variable configuration |
| `export CPU_AFFINITY_CONF=1` to enable coarse-grained CPU affinity | Environment variable configuration |
| `export COMBINED_ENABLE=1` to optimize non-contiguous operator combinations | Environment variable configuration |
| Replacing the `NpuFusedAdamW` fused optimizer | configs/bevdet/bevdet-r50.py |
| Replacing the `bev_pool_v3` fused operator | mmdet3d/models/necks/view_transformer.py |
| Replacing the `npu_gaussian` fused operator | mmdet3d/models/dense_heads/centerpoint_head.py |

### Model Migration Optimization

#### Implementation

Reference implementation:

```python
url=https://github.com/HuangJunJie2017/BEVDet.git
commit_id=58c2587a8f89a1927926f0bdb6cde2917c91a9a5
```

Implementation adapted for the Ascend AI processor:

```python
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/BEVDet
```

#### Dependency Installation

1. Autonomous driving models are currently configured mainly with PyTorch 2.1. Using PyTorch 2.1 as an example, first install the model-related dependencies. You can create a `requirements.txt` file and add the following dependencies and versions:

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

    Install them with `pip install -r requirements.txt`.

2. Install the third-party `mmcv` library. Currently, version 1.7.2 supports NPUs. Therefore, you need to download the code manually and compile it from source:

    ```python
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    ```

    To adapt to PyTorch 2.x, modify line 159 in `mmcv/mmcv/parallel/distributed.py`. The original code is:

    ```python
    module_to_run = self._replicated_tensor_module if self. _use_replicated_tensor_module else self.module
    ```

    Replace it with:

    ```python
    module_to_run = self.module
    ```

    Enter the `mmcv` root directory and compile and install `mmcv` with the following command:

    ```python
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
    ```

#### Migration Adaptation

1. First, download the official model source code and specify the commit ID:

    ```python
    git clone https://github.com/HuangJunJie2017/BEVDet.git
    cd BEVDet
    git checkout 58c2587a8f89a1927926f0bdb6cde2917c91a9a5
    ```

2. Add automatic migration code to `tools/train.py` and `tools/test.py`:

    ```python
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    ```

3. Adapt the DDP-related calls to `NPUDataParallel` and `NPUDistributedDataParallel`. Modify `tools/test.py`:

    ```python
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    ```

    Change it to:

    ```python
    from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
    ```

    Then change the original code:

    ```python
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    …
    model = MMDistributedDataParallel(
    model.cuda(),
    ```

    to:

    ```python
    model = NPUDataParallel(model.npu(), device_ids=cfg.gpu_ids)
    …
    model = NPUDistributedDataParallel(
    model.npu(),
    ```

    Modify `mmdet3d/apis/train.py` and change the original code:

    ```python
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    ```

    to:

    ```python
    from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
    ```

    Then change the corresponding code:

    ```python
    model = MMDistributedDataParallel( and model = MMDataParallel(
    ```

    to:

    ```python
    model = NPUDistributedDataParallel( and model = NPUDataParallel(
    ```

4. To adapt to PyTorch 2.1, add the `--local-rank` parameter. In `tools/train.py` and `tools/test.py`, the original code is:

    ```python
    parser.add_argument('--local_rank', type=int, default=0)
    ```

    Add the following before it:

    ```python
    parser.add_argument('--local-rank', type=int, default=0)
    ```

5. Remove the CUDA-related code. In `mmdet3d/models/detectors/init.py`, change the original code:

    ```python
    from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVDetTRT, BEVStereo4D
    ```

    to:

    ```python
    from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVStereo4D
    ```

    Comment out the original code:

    ```python
    from .dal import DAL
    ```

    Change the original code:

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

    to:

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

    Modify `mmdet3d/models/detectors/bevdet.py` and delete or comment out the following code:

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

6. For `mmdet_3d` version compatibility, modify `mmdet3d/__init__.py`. Change the original code:

    ```python
    mmcv_maximum_version = '1.7.0'
    ```

    to:

    ```python
    mmcv_maximum_version = '1.7.2'
    ```

    Comment out or delete the following original code:

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

7. The original `bev_pool_v2` operator in the source code uses CUDA code. You need to replace it with `bev_pool_v3` from the Driving SDK repository, which is more compatible and efficient. After installing Driving SDK, see the [README](../../../README_en.md).
    In `mmdet3d/models/necks/view_transformer.py`, delete the original code:

    ```python
    from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
    ```

    and add the import:

    ```python
    from mx_driving.point import bev_pool_v3
    ```

    In the `voxel_pooling_v2` function of the `LSSViewTransformer` class, replace the original code:

    ```python
    bev_feat = bev_pool_v2(depth, feat, ranks_depth,
            ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts,
            interval_lengths)
    ```

    with:

    ```python
    bev_feat = bev_pool_v3(depth, feat, ranks_depth,
            ranks_feat, ranks_bev,
            bev_feat_shape)
    ```

    In the `view_transform_core` function of the `LSSViewTransformer` class, replace the original code:

    ```python
    bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
            self.ranks_feat, self.ranks_bev,
            bev_feat_shape, self.interval_starts,
            self.interval_lengths)
    ```

    with:

    ```python
    bev_feat = bev_pool_v3(depth, feat, self.ranks_depth,
            self.ranks_feat, self.ranks_bev,
            bev_feat_shape)
    ```

8. `matmul` on the NPU does not support tensors with more than six dimensions. Therefore, you need to adapt the code. Replace the `get_lidar_coor` function in the `LSSViewTransformer` class in `mmdet3d/models/necks/view_transformer.py` with the following:

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

        # Post-transformation
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

#### Model Optimization

1. Replace the fused optimizer. In `configs/bevdet/bevdet-r50.py`, the original code is:

    ```python
    optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
    ```

    Change it to:

    ```python
    optimizer = dict(type='NpuFusedAdamW', lr=2e-4, weight_decay=1e-07)
    ```

2. Replace the `npu_gaussian` fused operator. Add the following import to `mmdet3d/models/dense_heads/centerpoint_head.py`:

    ```python
    from mx_driving import npu_gaussian
    ```

    Replace the `get_targets_single` function to reduce many small operator launches and improve computation efficiency:

    ```python
    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # Reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                feature_map_size[0]))

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                                dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                                dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            temp_classes = task_classes[idx]
            temp_boxes = task_boxes[idx]
            center_int, radius, mask, ind, anno_box = npu_gaussian(temp_boxes,
                                                                self.train_cfg['out_size_factor'],
                                                                self.train_cfg['gaussian_overlap'],
                                                                self.train_cfg['min_radius'],
                                                                voxel_size[0],
                                                                voxel_size[1],
                                                                pc_range[0],
                                                                pc_range[1],
                                                                feature_map_size[0],
                                                                feature_map_size[1],
                                                                self.norm_bbox,
                                                                self.with_velocity)
            for k in range(num_objs):
                cls_id = temp_classes[k] - 1

                width = temp_boxes[k][3]
                length = temp_boxes[k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    draw_gaussian(heatmap[cls_id], center_int[k], radius[k].item())
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)

        return heatmaps, anno_boxes, inds, masks
    ```
