# npu_gaussian

## Interface Prototype

```python
mx_driving.npu_gaussian(Tensor boxes, int out_size_factor, float gaussian_overlap, int min_radius, float voxel_size_x, float voxel_size_y, float pc_range_x,
    float pc_range_y, int feature_map_size_x, int feature_map_size_y, bool norm_bbox, bool with_velocity, bool flip_angle, int max_objs) -> (Tensor center_int, Tensor radius, Tensor mask, Tensor ind, Tensor anno_box)
```

## Description

Implements part of the `get_targets_single` function in the `centerpoint_head.py` script.

## Parameters

- `boxes` (Tensor): 3D bounding box information for each object. The data type is `float32`, and the shape is `[Num_objs, W]`.
- `out_size_factor` (int): Feature map downsampling factor.
- `gaussian_overlap` (float): Used to control the Gaussian radius calculation.
- `min_radius` (int): Minimum Gaussian radius.
- `voxel_size_x` (float): Voxel size in the x direction.
- `voxel_size_y` (float): Voxel size in the y direction.
- `pc_range_x` (float): Point cloud range in the x direction.
- `pc_range_y` (float): Point cloud range in the y direction.
- `feature_map_size_x` (int): Size of the BEV feature map in the x direction.
- `feature_map_size_y` (int): Size of the BEV feature map in the y direction.
- `norm_bbox` (bool): Whether to normalize the 3D bounding box parameters.
- `with_velocity` (bool): Whether to include velocity information in the object detection task.
- `flip_angle` (bool): Whether to flip the sine and cosine results in the output.
- `max_objs` (int): Upper limit on the number of `boxes` to process. In the BEVDet model, this value is 500.

## Returns

- `center_int` (Tensor): Calculated center coordinates. The data type is `int32`, and the shape is `[minObjs, 2]`.
- `radius` (Tensor): Calculated Gaussian radius. The data type is `int32`, and the shape is `[minObjs]`.
- `mask` (Tensor): Calculated mask for the eligible `boxes`. The data type is `uint8`, and the shape is `[max_objs]`.
- `ind` (Tensor): Calculated offsets of the centers of the eligible `boxes`. The data type is `int64`, and the shape is `[max_objs]`.
- `anno_box` (Tensor): Calculated 3D bounding box attributes. The data type is `float32`, and the shape is `[max_objs, 10]`.

## Constraints

1. If this operator is adapted to the BEVDet model, `W` must be 9.
2. All parameters must remain consistent with the model configuration.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import npu_gaussian

out_size_factor = 8
gaussian_overlap = 0.1
min_radius = 2
voxel_size_x = 0.1
voxel_size_y = 0.1
pc_range_x = -51.2
pc_range_y = -51.2
feature_map_size_x = 128
feature_map_size_y = 128
norm_bbox = True
with_velocity = True
flip_angle = True
num_objs = 100
boxes = -50 + 100 * torch.rand((num_objs, 9), dtype=torch.float32).npu()
output = npu_gaussian(boxes, out_size_factor, gaussian_overlap, min_radius, voxel_size_x, voxel_size_y, pc_range_x, pc_range_y, feature_map_size_x, feature_map_size_y, norm_bbox, with_velocity, flip_angle, num_objs)
```
