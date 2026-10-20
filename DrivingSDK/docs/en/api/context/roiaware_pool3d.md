# roiaware_pool3d

## Interface Prototype

```python
mx_driving.roiaware_pool3d(Tensor rois, Tensor pts, Tensor pts_feature,
                    Union[int, tuple] out_size, int max_pts_per_voxel, int mode) -> Tensor
```

## Description

Pools the input point cloud features inside Region of Interest (RoI) boxes.

## Parameters

- `rois` (Tensor): Coordinates and dimensions of the input RoI boxes. The data type is `float32/float16`. The shape is `[Roi_num, 7]`.
- `pts` (Tensor): Coordinates of the input points. The data type is `float32/float16`. The shape is `[Pts_num, 3]`.
- `pts_feature` (Tensor): Feature vectors of the input points. The data type is `float32/float16`. The shape is `[Pts_num, Channels]`.
- `out_size` (Union): Size of the voxels inside the output RoI boxes. The data type is `int` or `tuple`. The shape is `[out_x, out_y, out_z]`.
- `max_pts_per_voxel` (int): Maximum number of points in each voxel. The data type is `int`.
- `mode` (int): Pooling method. 0 indicates max pooling, and 1 indicates average pooling. The data type is `int`.

## Returns

- `pooled_features` (Tensor): Pooled RoI box features. The data type is `float32/float16`. The shape is `[Roi_num, out_x, out_y, out_z, Channels]`.

## Constraints

- `Roi_num <= 100`.
- `Pts_num <= 1000`.
- `Channels <= 1024`.
- `2 <= max_pts_per_voxel <= 256` and `max_pts_per_voxel <= Pts_num`.
- `out_x`, `out_y`, and `out_z` are all `<= 16`.
- The backward pass has the same constraints.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import math
import torch_npu
from mx_driving import roiaware_pool3d
import numpy as np

out_size = (5, 5, 5)
max_pts_per_voxel = 128
mode = 1

N = 40
npoints = 1000
channels = 1024

xyz_coor = np.random.uniform(-1, 1, size = (N, 3)).astype(np.float32)
xyz_size_num = np.random.uniform(5, 50, size = (1, 3))
xyz_size = (xyz_size_num * np.ones((N, 3))).astype(np.float32)
angle = np.radians(np.random.randint(0, 360, size = (N , 1))).astype(np.float32)

rois = np.concatenate((xyz_coor, xyz_size), axis=1)
rois = np.concatenate((rois, angle), axis=1)
rois = torch.tensor(rois).npu()

pts = np.random.uniform(-5, 5, size = (npoints, 3)).astype(np.float32)
pts = torch.tensor(pts).npu()
pts_feature = np.random.uniform(-1, 1, size=(npoints, channels)).astype(np.float32)
pts_feature = torch.tensor(pts_feature).npu()

pooled_features_npu = roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_per_voxel, mode)

pooled_features_npu.requires_grad = True
grad_out_tensor = torch.ones_like(pooled_features_npu)
pooled_features_npu.backward(grad_out_tensor)
```
