# nms3d_normal

## Interface Prototype

```python
mx_driving.nms3d_normal(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor
```

## Description

Performs 3D non-maximum suppression.

## Parameters

- `boxes` (Tensor): Box tensor. The data type is `float32` or `float16`. The shape is `[N, 7]`. The seven values represent `x`, `y`, `z`, `x_size`, `y_size`, `z_size`, and `rz`.
- `scores` (Tensor): Score tensor. The data type is `float32` or `float16`. The shape is `[N]`.
- `iou_threshold` (float): Intersection over Union (IoU) threshold.

## Returns

- `output` (Tensor): Tensor of boxes after NMS. The data type is `int32`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import nms3d_normal
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = nms3d_normal(boxes, scores, 0.5)
```
