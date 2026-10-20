# furthest_point_sample_with_dist

## Interface Prototype

```python
mx_driving.furthest_point_sample_with_dist(Tensor points, int num_points) -> Tensor
```

## Description

Same as `furthest_point_sampling`, but the input is slightly different.

## Parameters

- `points` (Tensor): Pairwise distance matrix for the point cloud. The data type is `float32`. The shape is `[B, N, N]`. `B` is the batch size and `N` is the number of points.
- `num_points` (int): Number of points to sample.

## Returns

- `output` (Tensor): Sampled point cloud data. The data type is `float32`. The shape is `[B, num_points]`.

## Constraints

1. Performance is better when `N` is large.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
```
