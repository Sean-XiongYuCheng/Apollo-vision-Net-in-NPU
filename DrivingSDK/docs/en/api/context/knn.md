# knn

## Interface Prototype

```python
mx_driving.knn(int k, Tensor xyz, Tensor center_xyz, bool Transposed) -> Tensor
```

## Description

Finds the `k` nearest points in `xyz` for each point in `center_xyz` in the corresponding batch, and returns their indices.

## Parameters

- `xyz` (Tensor): Point data that represents 3D coordinates `(x, y, z)`. The data type is `float32`. The shape is `[B, N, 3]` when `Transposed=False`, or `[B, 3, N]` when `Transposed=True`. `B` is the batch size, and `N` is the number of points.
- `center_xyz` (Tensor): Point data that represents 3D coordinates `(x, y, z)`. The data type is `float32`. The shape is `[B, npoint, 3]` when `Transposed=False`, or `[B, 3, npoint]` when `Transposed=True`. `B` is the batch size, and `npoint` is the number of points.
- `k` (int): Number of sampled points.
- `Transposed` (bool): Whether the input needs to be transposed.

## Returns

- `idx` (Tensor): Sampled index data. The data type is `int32`. The shape is `[B, k, npoint]`.

## Constraints

1. `k` must be greater than 0 and less than 100.
2. The distance between any point in each batch of `xyz` and any point in the corresponding batch of `center_xyz` must be within `1e10f`.
3. Both `xyz` and `center_xyz` must be 3D shapes. When `Transposed=True`, dimension 1 of the `xyz` and `center_xyz` shapes must be 3. When `Transposed=False`, dimension 2 of the `xyz` and `center_xyz` shapes must be 3.
4. When distances are equal, sorting is unstable. Therefore, the distance precision meets the requirement, but the index precision can differ, and the result cannot fully align with competing products.
5. Performance is better when `N` is large.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import knn
xyz = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
center_xyz = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
idx = knn(2, xyz, center_xyz, False)
```
