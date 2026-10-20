# npu_rotated_iou[beta]

## Interface Prototype

```python
mx_driving.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```

## Description

Computes the Intersection over Union (IoU) of rotated bounding boxes.

## Parameters

- `self` (Tensor): Input bounding box tensor. The data type is `float32, float16`. The shape is `[B, N, 5]`.
- `query_boxes` (Tensor): Query bounding box tensor. The data type is `float32, float16`. The shape is `[B, M, 5]`.
- `trans` (bool): Whether to apply a coordinate transform. The default value is `False`. When the value is `True`, the format is `xyxyt`. When the value is `False`, the format is `xywht`, where `t` is the angle in degrees.
- `is_cross` (bool): When the value is `True`, this parameter computes the result for every pair of boxes across the two bounding box sets. When the value is `False`, it computes the result only for aligned bounding boxes.
- `mode` (int): IoU computation mode. The default value is `0`. When the value is `0`, it computes `IoU`. When the value is `1`, it computes `IoF`.
- `v_threshold` (float): Tolerance threshold for vertex determination.
- `e_threshold` (float): Tolerance threshold for edge intersection determination.

## Returns

- `output` (Tensor): IoU tensor. The data type is `float32, float16`. When `is_cross` is `True`, the shape is `[B, N, M]`. Otherwise, the shape is `[B, N]`.

## Constraints

- The value range of `mode` is `{0, 1}`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Example

```python
import torch, torch_npu
import numpy as np
from mx_driving import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, False, 0, True, 1e-5, 1e-5)
```
