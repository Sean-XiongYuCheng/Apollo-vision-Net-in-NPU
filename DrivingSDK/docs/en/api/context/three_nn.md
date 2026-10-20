# three_nn

## Interface Prototype

```python
mx_driving.three_nn(Tensor target, Tensor source) -> (Tensor dist, Tensor idx)
```

## Description

For each point in `target`, find the three nearest points in the corresponding batch of `source`, and return their distances and indices.

## Parameters

- `target` (Tensor): Point data that represents `(x, y, z)` 3D coordinates. The data type is `float32/float16`. The shape is `[B, npoint, 3]`, where `B` is the batch size and `npoint` is the number of points.
- `source` (Tensor): Point data that represents `(x, y, z)` 3D coordinates. The data type is `float32/float16`. The shape is `[B, N, 3]`, where `B` is the batch size and `N` is the number of points.

## Returns

- `dist` (Tensor): Distance data for the three nearest points. The data type is `float32/float16`. The shape is `[B, npoint, 3]`.
- `idx` (Tensor): Index data for the three nearest points. The data type is `int32/int64`. The shape is `[B, npoint, 3]`.

## Constraints

1. The shapes of `source` and `target` must be 3D, and the second dimension of both shapes must be 3.
2. When distances are equal, sorting is unstable. Therefore, although the distance precision meets the requirement, the index precision may be incorrect, which prevents the result from fully aligning with competing products.
3. Performance is better when `N` is larger.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import three_nn
source = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
target = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
dist, idx = three_nn(target, source)
```
