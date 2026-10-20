# nms3d

## Interface Prototype

```python
mx_driving.nms3d(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor
```

## Description

Performs 3D non-maximum suppression in the BEV view and removes boxes whose 3D Intersection over Union (IoU) exceeds the threshold.

## Parameters

- `boxes` (Tensor): Box tensor. The data type is `float32` or `float16`. The shape is `[N, 7]`. The 7 values represent `x`, `y`, `z`, `x_size`, `y_size`, `z_size`, and `rz`.
- `scores` (Tensor): Score tensor. The data type is `float32` or `float16`. The shape is `[N]`.
- `iou_threshold` (float): IoU threshold.

## Returns

- `output` (Tensor): NMS output box tensor. The data type is `int32`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import nms3d
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = nms3d(boxes, scores, 0.5)
```
