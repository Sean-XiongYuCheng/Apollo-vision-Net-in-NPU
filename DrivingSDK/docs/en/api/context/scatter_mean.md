# scatter_mean

## Interface Prototype

```python
mx_driving.scatter_mean(Tensor src, Tensor indices, Tensor out=None, int dim=0, int dim_size=None) -> Tensor
```

## Description

Groups elements in the input tensor `src` by the indices in `indices` along the specified `dim` dimension, computes the mean of each group, and returns the result.

## Parameters

- `src` (Tensor): Source tensor. The data type is `float32`.
- `indices` (Tensor): Index tensor. The data type is `int32`.
- `out` (Tensor): Tensor to be updated. The data type is `float32`. This is an optional input and defaults to `None`. When `out` is not `None`, elements in `out` participate in the mean calculation.
- `dim` (int): Dimension to use for grouping and mean calculation. The data type is `int32`. This is an optional input and defaults to `0`.
- `dim_size` (int): Length of the output tensor along the `dim` dimension. The data type is `int32`. This is an optional input and defaults to `None`. This parameter takes effect only when `out` is `None`.

## Returns

- `out` (Tensor): Averaged tensor. The data type is `float32`.

## Constraints

- The number of dimensions of `indices` must be less than or equal to that of `src`, and the length of each dimension must match the corresponding dimension of `src`.
- The values in `indices` must be valid nonnegative index values. When `out` or `dim_size` is not `None`, the values in `indices` must be valid indices along the `dim` dimension of the output tensor.
- The number of dimensions of `out` must be the same as that of `src`, and the length of every dimension except `dim` must match that of `src`.
- `dim` cannot exceed the number of dimensions of `indices`.
- `dim_size` must be a valid nonnegative length value.
- `src` and `out` do not support `inf`, `-inf`, or `nan`.
- The operator performs better in both forward and backward passes when the tail block is larger. It does not perform well when the tail block is very small. Here, the tail block refers to the size of the `N` trailing dimensions of `src`, where `N = src.dim() - indices.dim()`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import scatter_mean
src = torch.randn(4, 5, 6).to(torch.float)
indices = torch.randint(5, (4, 5)).to(torch.int32)
dim = 0
src.requires_grad = True
out = scatter_mean(src.npu(), indices.npu(), None, dim)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
