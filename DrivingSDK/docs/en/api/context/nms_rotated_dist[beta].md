# nms_rotated_dist[beta]

## Interface Prototype

```python
mx_driving.nms_rotated_dist(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor
```

## Description

Performs rotated-box non-maximum suppression (NMS), which determines whether rotated boxes overlap based on the projection distance and removes redundant boxes.

After sorting the boxes in descending order of `scores`, the function takes each box in turn as the reference and projects the distance between the center of each subsequent box and the center of the current box onto the local coordinate system of the current box (along the `angle` direction and the direction perpendicular to it). When the projected distance exceeds the length and width range of the current box, the function considers the two boxes non-overlapping and retains them. Otherwise, it suppresses the box with the lower score. Compared with NMS based on exact IoU, this method approximates the overlap determination using the projection distance and incurs lower computational cost. Therefore, it is suitable for rotated-box detection post-processing scenarios where accuracy requirements are not extreme and performance is prioritized.

## Parameters

- `boxes` (Tensor): Rotated-box tensor. The data type is `float32`. The shape is `[N, 5]`. The five values represent `x_center`, `y_center`, `width`, `height`, and `angle`. `x_center` and `y_center` indicate the coordinates of the box center. `width` and `height` indicate the box length and width. `angle` indicates the rotation angle in radians. The input must be on the NPU.
- `scores` (Tensor): Score tensor. The data type is `float32`. The shape is `[N]`. The input must be on the NPU.
- `iou_threshold` (float): Suppression threshold. Because the overlap determination uses the projection distance, the function suppresses the boxes determined to overlap when the threshold is less than 0.5. When the threshold is greater than or equal to 0.5, it performs no suppression.

## Returns

- `output` (Tensor): Index tensor of the boxes retained after NMS. The data type is `int64`. The shape is `[K]`, where `K` is the number of retained boxes. The indices are returned in descending order of `scores`.

## Constraints

- `boxes` and `scores` must be on an NPU device. Otherwise, the function raises a `ValueError`.
- `boxes` supports only the `float32` data type.
- The last dimension of `boxes` must be `5`.
- The 0th dimension (`N`) of `boxes` and `scores` must match.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Example

```python
import torch, torch_npu
from mx_driving import nms_rotated_dist
boxes = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5],
                      [0.0, 2.0, 2.0, 5.0, 0.3]], dtype=torch.float32).npu()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).npu()
keep = nms_rotated_dist(boxes, scores, 0.2)
```
