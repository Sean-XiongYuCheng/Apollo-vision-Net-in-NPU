# furthest_point_sampling

## Interface Prototype

```python
mx_driving.furthest_point_sampling(Tensor points, int num_points) -> Tensor
```

## Description

Performs furthest point sampling on point cloud data.

## Parameters

- `points` (Tensor): Point cloud data. The data type is `float32`, `float16`, or `bfloat16`. The shape is `[B, N, 3]`. `B` is the batch size, `N` is the number of points, and 3 represents `x`, `y`, and `z`.
- `num_points` (int): Number of points to sample.

## Returns

- `output` (Tensor): Sampled point indices. The data type is `int32`. The shape is `[B, num_points]`.

## Constraints

- The total size of the `points` input, with shape `[B, N, 3]` (`B x N x 3`), must not exceed 383166.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sampling(points, 2)
```
