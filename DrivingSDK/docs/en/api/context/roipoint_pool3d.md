# roipoint_pool3d

## Interface Prototype

```python
mx_driving.roipoint_pool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
```

## Description

Encodes the geometry-specific features of each 3D proposal.

## Parameters

- `num_sampled_points` (int): Number of sampled points. It must be a positive integer.
- `points` (Tensor): Point tensor. The data type is `float32` or `float16`. The shape is `[B, N, 3]`, where 3 denotes `x`, `y`, and `z`.
- `point_features` (Tensor): Point feature tensor. The data type is `float32` or `float16`. The shape is `[B, N, C]`.
- `boxes3d` (Tensor): Box tensor. The data type is `float32` or `float16`. The shape is `[B, M, 7]`, where 7 denotes `x`, `y`, `z`, `x_size`, `y_size`, `z_size`, and `rz`.

## Returns

- `pooled_features` (Tensor): Feature tensor for points inside the boxes. The data type is `float32` or `float16`. The shape is `[B, M, num_sampled_points, 3 + C]`.
- `pooled_empty_flag` (Tensor): Empty-flag tensor indicating that no points fall inside a box. The data type is `int32`. The shape is `[B, M]`.

## Constraints

- The data types of `points`, `point_features`, and `boxes3d` must match, and `B` must also match.
- `num_sampled_points` must be less than or equal to `N`.
- When the data type is `float32`, you are advised to keep `B` below 96, `N` at or below 2048, `M` at or below 48, `num_sampled_points` at or below 48, and `C` at or below 8. Slightly exceeding one shape value has no effect. However, if all shape values exceed the recommended values, the operator fails.
- When the data type is `float16`, you are advised to keep `B` below 96, `N` at or below 3192, `M` at or below 60, `num_sampled_points` at or below 60, and `C` at or below 8. Slightly exceeding one shape value has no effect. However, if all shape values exceed the recommended values, the operator fails.
- The larger `N` and `M` are, the more performance degrades. Therefore, you are advised to keep `N` below 600 times `M`. Otherwise, performance may be lower than `0.1x A100`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import roipoint_pool3d
num_sampled_points = 1
points = torch.tensor([[[1, 2, 3]]], dtype=torch.float).npu()
point_features = points.clone()
boxes3d = torch.tensor([[[1, 2, 3, 4, 5, 6, 1]]], dtype=torch.float).npu()
pooled_features, pooled_empty_flag = roipoint_pool3d(num_sampled_points, points, point_features, boxes3d)
```
