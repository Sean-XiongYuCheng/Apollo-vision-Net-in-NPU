# boxes_iou_bev

## Interface Prototype

```python
mx_driving.boxes_iou_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```

## Description

Calculate the Intersection over Union (IoU) of two bounding boxes in the bird's eye view (BEV).

## Parameters

- `boxes_a` (Tensor): First set of bounding boxes. The data type is `float32`. The shape is `[M, 7]`. The seven values represent `x_center`, `y_center`, `z_center`, `dx`, `dy`, `dz`, and `angle`. `x_center`, `y_center`, and `z_center` are the center coordinates of the box. `dx`, `dy`, and `dz` are the box length, width, and height. `angle` is the rotation angle in radians.
- `boxes_b` (Tensor): Second set of bounding boxes. The data type is `float32`. The shape is `[N, 7]`. The seven values represent `x_center`, `y_center`, `z_center`, `dx`, `dy`, `dz`, and `angle`. `x_center`, `y_center`, and `z_center` are the center coordinates of the box. `dx`, `dy`, and `dz` are the box length, width, and height. `angle` is the rotation angle in radians.

## Returns

- `ious` (Tensor): Tensor that contains the IoU values for the two sets of bounding boxes. The data type is `float32`. The shape is `[M, N]`.

## Constraints

- `angle` is in [-pi, pi].
- The product of `M` and `N` does not exceed 900 million.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import boxes_iou_bev
boxes_a = torch.tensor([[1.0, 1.0, 1.0, 3.0, 4.0, 1.0, 0.5]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[0.0, 2.0, 1.0, 2.0, 5.0, 1.0, 0.3]], dtype=torch.float32).npu()
ious = boxes_iou_bev(boxes_a, boxes_b)
```
