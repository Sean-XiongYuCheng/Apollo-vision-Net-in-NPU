# box_iou_quadri

## Interface Prototype

```python
mx_driving.box_iou_quadri(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False) -> Tensor
```

## Description

Computes the Intersection over Union (IoU) of two bounding boxes.

## Parameters

- `boxes_a` (Tensor): First set of bounding boxes. The data type is `float32`. The shape is `[M, 8]`. The eight values represent `x1`, `y1`, `x2`, `y2`, `x3`, `y3`, `x4`, and `y4`, which are the horizontal and vertical coordinates of the four vertices of the box.
- `boxes_b` (Tensor): Second set of bounding boxes. The data type is `float32`. The shape is `[N, 8]`. The eight values represent `x1`, `y1`, `x2`, `y2`, `x3`, `y3`, `x4`, and `y4`, which are the horizontal and vertical coordinates of the four vertices of the box.
- `mode` (str): When the value is `"iou"`, the operator computes the IoU (intersection over union). When the value is `"iof"`, the operator computes the IoF (intersection over foreground).
- `aligned` (bool): When the value is `True`, the operator computes results only for paired boxes. When the value is `False`, the operator computes results for every pair of boxes.

## Returns

- `ious` (Tensor): Tensor that contains the IoU when `mode="iou"` or the IoF when `mode="iof"` for the two sets of bounding boxes. The data type is `float32`. The shape is `[M]` when `aligned=True` or `[M, N]` when `aligned=False`.

## Constraints

- The value of `mode` must be one of `{'iou', 'iof'}`.
- When `aligned=False`, the product of the number of boxes in `boxes_a` (`M`) and the number of boxes in `boxes_b` (`N`) must not exceed 900 million.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import box_iou_quadri
boxes_a = torch.tensor([[7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]], dtype=torch.float32).npu()
ious = box_iou_quadri(boxes_a, boxes_b, mode="iou", aligned=False)
```
