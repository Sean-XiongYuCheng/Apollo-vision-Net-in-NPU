# box_iou_rotated[beta]

## Interface Prototype

```python
mx_driving.box_iou_rotated(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False, bool clockwise=True) -> Tensor
```

## Description

Computes the Intersection over Union (IoU) between two bounding boxes.

## Parameters

- `boxes_a` (Tensor): First set of bounding boxes. The data type is `float32`. The shape is `[M, 5]`. The five values represent `x_center`, `y_center`, `dx`, `dy`, and `angle`. `x_center` and `y_center` indicate the coordinates of the box center. `dx` and `dy` indicate the box length and width. `angle` indicates the rotation angle in radians.
- `boxes_b` (Tensor): Second set of bounding boxes. The data type is `float32`. The shape is `[N, 5]`. The five values represent `x_center`, `y_center`, `dx`, `dy`, and `angle`. `x_center` and `y_center` indicate the coordinates of the box center. `dx` and `dy` indicate the box length and width. `angle` indicates the rotation angle in radians.
- `mode` (str): When the value is `"iou"`, computes the IoU. When the value is `"iof"`, computes the IoF (intersection over foreground).
- `aligned` (bool): When the value is `True`, computes results only for paired boxes. When the value is `False`, computes results for each pair of boxes.
- `clockwise` (bool): Whether the rotation angle `angle` uses the clockwise direction as the positive direction.

## Returns

- `ious` (Tensor): Tensor that contains the IoU of the two sets of bounding boxes when `mode="iou"`, or the IoF when `mode="iof"`. The data type is `float32`. The shape is `[M]` when `aligned=True` or `[M, N]` when `aligned=False`.

## Constraints

- The value range of `mode` is `{'iou', 'iof'}`.
- When `aligned=False`, the product of the number of boxes in `boxes_a` (`M`) and the number of boxes in `boxes_b` (`N`) does not exceed 900 million.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import box_iou_rotated
boxes_a = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[0.0, 2.0, 2.0, 5.0, 0.3]], dtype=torch.float32).npu()
ious = box_iou_rotated(boxes_a, boxes_b, mode="iou", aligned=False, clockwise=True)
```
