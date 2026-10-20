# points_in_box

## Interface Prototype

```python
mx_driving.points_in_box(Tensor boxes, Tensor points) -> Tensor
```

## Description

Determines whether points lie inside boxes.

## Parameters

- `boxes` (Tensor): Box tensor. The data type is `float32`. The shape is `[B, M, 7]`. 7 represents `x`, `y`, `z`, `x_size`, `y_size`, `z_size`, and `rz`.
- `points` (Tensor): Point tensor. The data type is `float32`. The shape is `[B, N, 3]`. 3 represents `x`, `y`, and `z`.

## Returns

- `boxes_idx_of_points` (Tensor): Index tensor of the boxes containing the points. The data type is `int32`. The shape is `[B, N]`.

## Constraints

- `boxes` and `points` must share the same `B`, and `B` can only be 1.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import points_in_box
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32).npu()
out = points_in_box(boxes, points)
```
