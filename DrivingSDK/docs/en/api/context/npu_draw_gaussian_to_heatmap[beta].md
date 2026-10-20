# npu_draw_gaussian_to_heatmap[beta]

## Interface Prototype

```python
mx_driving.npu_draw_gaussian_to_heatmap(Tensor mask, Tensor cur_class_id, Tensor center_int_trans, Tensor radius, int feature_map_size_x, int feature_map_size_y, int num_classes) -> (Tensor heatmap)
```

## Description

Implements part of the `get_targets_single` function in the `centerpoint_head.py` script. Use it together with `npu_gaussian`.

## Parameters

- `mask` (Tensor): 3D bounding box information for each object. The data type is `uint8`, and the shape is `[maxObjs, W]`.
- `cur_class_id` (Tensor): Heatmap index of the corresponding object. The data type is `int32`, and the shape is `[Num_objs]`.
- `center_int_trans` (Tensor): Integer coordinates of the target center point. The data type is `int32`, and the shape is `[2, minObjs]`.
- `radius` (Tensor): Gaussian radius computed by `npu_gaussian`. The data type is `int32`.
- `feature_map_size_x` (int): Size of the BEV feature map in the x direction.
- `feature_map_size_y` (int): Size of the BEV feature map in the y direction.
- `num_classes` (int): Number of heatmap classes.

## Returns

- `heatmap` (Tensor): Computed heatmap. The data type is `float32`, and the shape is `[max_objs]`.

## Constraints

1. You must use it together with the `npu_gaussian` API.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import npu_gaussian, npu_draw_gaussian_to_heatmap

out_size_factor = 8
gaussian_overlap = 0.1
min_radius = 2
voxel_size_x = 0.1
voxel_size_y = 0.1
pc_range_x = -51.2
pc_range_y = -51.2
feature_map_size_x = 128
feature_map_size_y = 128
num_classes = 1
norm_bbox = True
with_velocity = True
num_objs = 100
boxes = -50 + 100 * torch.rand((num_objs, 9), dtype=torch.float32).npu()
cur_class_id = torch.ones(num_objs, dtype=torch.int32).npu()

output = npu_gaussian(boxes, out_size_factor, gaussian_overlap, min_radius, voxel_size_x, voxel_size_y, pc_range_x, pc_range_y, feature_map_size_x, feature_map_size_y, norm_bbox, with_velocity)
center_int_trans = output[0].t().contiguous()
radius = output[1]
mask = output[2]
heatmap = npu_draw_gaussian_to_heatmap(mask, cur_class_id, center_int_trans, radius, feature_map_size_x, feature_map_size_y, num_classes)
```
