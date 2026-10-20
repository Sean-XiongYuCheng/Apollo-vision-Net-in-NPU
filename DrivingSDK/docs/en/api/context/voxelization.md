# voxelization

## Interface Prototype

```python
mx_driving.Voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True, str layout="XYZ") -> Tensor
```

## Description

Transforms point cloud data into voxels.

## Parameters

- `points` (Tensor): Point cloud data. The data type is `float32`. The shape is `[N, F]`. `N` is the number of points, and `F` is the feature dimension of each point. `N > 0` and `F >= 3`.
- `voxel_size` (List[float]): Voxel size. The data type is `float32`. The shape is `[3]`. The three values represent `x`, `y`, and `z`.
- `coors_range` (List[float]): Voxel range. The data type is `float32`. The shape is `[6]`. The six values represent `x_min`, `y_min`, `z_min`, `x_max`, `y_max`, and `z_max`.
- `max_points` (int): Maximum number of points per voxel. The default value is -1. `max_points` <= 1000.
- `max_voxels` (int): Maximum number of voxels. The default value is -1.
- `deterministic` (bool): Whether to use deterministic behavior. The default value is `True`.
- `layout` (str): Sorting order of voxel coordinates. The default value is `XYZ`. Use `ZYX` when replacing the MMCV operator, and use `XYZ` when replacing the mmdetection3d operator.

## API Semantics

When `max_points` is -1 or `max_voxels` is -1, the operator performs `dynamic_voxelize`. Otherwise, it performs `hard_voxelize`.

## Returns

When `dynamic_voxelize` is performed:

- `coors` (Tensor): Voxel coordinates for each point. The data type is `int32`. The shape is `[N, 3]`.

When `hard_voxelize` is performed:

- `voxel_num` (int): Number of voxels after deduplication.
- `vox_points` (Tensor): Point cloud data for each voxel. The data type is `float32`. The shape is `[voxel_num, max_points, F]`.
- `coors` (Tensor): Voxel coordinates after deduplication. The data type is `int32`. The shape is `[voxel_num, 3]`.
- `num_points_per_voxel` (Tensor): Number of points in each voxel. The data type is `int32`. The shape is `[voxel_num]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import Voxelization
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 0, 9]
max_num_points = -1
max_voxels = 3500
vlz = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)
points = torch.rand(16, 3, dtype = torch.float32).npu()
results = vlz(points)
```
