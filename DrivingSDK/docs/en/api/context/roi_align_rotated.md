# roi_align_rotated

## Interface Prototype

```python
mx_driving.roi_align_rotated(Tensor feature_map, Tensor rois, float spatial_scale,
                             int sampling_ratio, int pooled_height, int pooled_width, bool aligned, bool clockwise) -> Tensor
```

## Description

Computes the Region of Interest (RoI) Align pooled feature map for rotated RoI boxes.

## Parameters

- `feature_map` (Tensor): Input feature map. The data type is `float32`. The shape is `[B, C, H, W]`.
- `rois` (Tensor): RoI tensor. The data type is `float32`. The shape is `[n, 6]`.
- `spatial_scale` (float): Scale factor for the RoI bounding boxes. The data type is `float32`.
- `sampling_ratio` (int): Sampling ratio. The data type is `int`. The value must be a non-negative integer.
- `pooled_height` (int): Pooled feature map height. The data type is `int`.
- `pooled_width` (int): Pooled feature map width. The data type is `int`.
- `aligned` (bool): Whether the boxes are aligned. The data type is `bool`. When `True`, the boxes are aligned. When `False`, the boxes are not aligned.
- `clockwise` (bool): Rotation direction of the rotated RoI boxes. The data type is `bool`. When `True`, the boxes rotate counterclockwise. When `False`, the boxes rotate clockwise.

## Returns

- `output` (Tensor): Pooled feature map. The data type is `float32`. The shape is `[n, C, pooled_height, pooled_width]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import math
import torch, torch_npu
import numpy as np
from mx_driving import roi_align_rotated

feature_map = torch.rand([1, 3, 16, 16])
feature_map.requires_grad = True

rois = torch.Tensor(8, 6)
rois[:, 0] = torch.randint(0, 1, (8,))
rois[:, 1].uniform_(0, 16)
rois[:, 2].uniform_(0, 16)
rois[:, 3].uniform_(0, 16)
rois[:, 4].uniform_(0, 16)
rois[:, 5].uniform_(0, math.pi)

output = roi_align_rotated(feature_map.npu(), rois.npu(), 1, 1, 7, 7, True, True)
output.backward(torch.ones_like(output))
```

## Notes

During bilinear interpolation sampling, when the sample point `x` is close to -1 or `W`, or `y` is close to -1 or `H`, platform differences and numerical errors may prevent the precision of that sample point from fully matching competing products.
During backward gradient propagation, atomic accumulation and the issue of large values overshadowing small values in floating-point arithmetic can cause fluctuations in the computed result for a point on the feature map when the gradient is propagated to that point more than 15,000 times. Therefore, its precision may not fully match competing products.
