# points_in_boxes_all

## Interface Prototype

```python
mx_driving.points_in_boxes_all(Tensor boxes, Tensor points) -> Tensor
```

## Description

Determines whether each point lies inside each box.

## Parameters

- `boxes` (Tensor): Box tensor. The data type is `float32`. The shape is `[B, M, 7]`. The seven values represent `x`, `y`, `z`, `x_size`, `y_size`, `z_size`, and `rz`.
- `points` (Tensor): Point tensor. The data type is `float32`. The shape is `[B, N, 3]`. The three values represent `x`, `y`, and `z`.

## Returns

- `boxes_idx_of_points` (Tensor): Tensor that indicates whether each point lies inside each box in the same batch. The data type is `int32`. The shape is `[B, N, M]`.

## Constraints

- The `B` dimension of `boxes` and `points` must be the same.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import points_in_boxes_all
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 5], [3, 4, 8]]], dtype=torch.float32).npu()
out = points_in_boxes_all(boxes, points)
```
