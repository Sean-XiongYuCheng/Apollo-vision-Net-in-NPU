# boxes_overlap_bev

## Interface Prototype

```python
mx_driving.boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```

## Description

Calculates the overlap area of two bounding boxes in the bird's eye view (BEV).

## Parameters

- `boxes_a` (Tensor): First set of bounding boxes. The data type is `float32`. The shape is `[M, 5]`. The 5 values represent `x1`, `y1`, `x2`, `y2`, and `angle`. `x1`, `y1`, `x2`, and `y2` represent the horizontal and vertical coordinates of the four vertices of the box. `angle` represents the rotation angle of the box in radians.
- `boxes_b` (Tensor): Second set of bounding boxes. The data type is `float32`. The shape is `[N, 5]`. The 5 values represent `x1`, `y1`, `x2`, `y2`, and `angle`. `x1`, `y1`, `x2`, and `y2` represent the horizontal and vertical coordinates of the four vertices of the box. `angle` represents the rotation angle of the box in radians.

## Returns

- `area_overlap` (Tensor): Tensor that contains the overlap area between the two sets of bounding boxes. The data type is `float32`. The shape is `[M, N]`.

## Constraints

- The value of `angle` must fall within [-pi, pi].
- The product of `M`, the number of `boxes_a` items, and `N`, the number of `boxes_b` items, must not exceed 900 million.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import boxes_overlap_bev
boxes_a = torch.tensor([[0, 0, 2, 2, 0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[1, 1, 3, 3, 0]], dtype=torch.float32).npu()
area_overlap = boxes_overlap_bev(boxes_a, boxes_b)
```
