# diff_iou_rotated_2d

## Interface Prototype

```python
mx_driving.diff_iou_rotated_2d(Tensor boxes_a, Tensor boxes_b) -> Tensor
```

## Description

Computes the Intersection over Union (IoU) between two rotated 2D object detection boxes and supports automatic differentiation.

## Parameters

- `boxes_a` (Tensor): First set of bounding boxes. The data type is `float32`. The shape is `[B, N, 5]`. Here, `B` indicates the batch size, `N` indicates the number of detection boxes in each batch, and the five values represent `x_center`, `y_center`, `dx`, `dy`, and `angle`. `x_center` and `y_center` represent the coordinates of the box center. `dx` and `dy` represent the box width and height. `angle` represents the rotation angle in radians.
- `boxes_b` (Tensor): Second set of bounding boxes. The data type is `float32`. The shape is `[B, N, 5]`. Here, `B` indicates the batch size, `N` indicates the number of detection boxes in each batch, and the five values represent `x_center`, `y_center`, `dx`, `dy`, and `angle`. `x_center` and `y_center` represent the coordinates of the box center. `dx` and `dy` represent the box width and height. `angle` represents the rotation angle in radians.

## Returns

- `ious` (Tensor): Tensor that contains the IoU values for the two sets of bounding boxes. The data type is `float32`. The shape is `[B, N]`.

## Constraints

- The value of `angle` must fall within [-pi, pi].
- `B` must fall within [1, 1024].
- `N` must fall within [1, 1024].

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import diff_iou_rotated_2d
boxes_a = torch.tensor([[[1.0, 1.0, 1.0, 3.0, 0.5]]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[[0.0, 2.0, 1.0, 2.0, 0.3]]], dtype=torch.float32).npu()
ious = diff_iou_rotated_2d(boxes_a, boxes_b)
```
