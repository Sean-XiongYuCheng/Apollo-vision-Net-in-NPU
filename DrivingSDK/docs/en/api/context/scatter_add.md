# scatter_add

## Interface Prototype

```python
mx_driving.scatter_add(Tensor src, Tensor indices, Tensor out=None, int dim=0, int dim_size=None) -> Tensor
```

## Description

Groups the elements in the input tensor `src` by the indices in `indices` along the specified `dim` dimension, sums each group, and writes the result to `out`.

## Parameters

- `src` (Tensor): Source tensor. The data type is `float32`.
- `indices` (Tensor): Index tensor. The data type is `int32`.
- `out` (Tensor): Updated tensor. The data type is `float32`. This is an optional input. The default value is `None`. When `out` is not `None`, its elements participate in the summation.
- `dim` (int): Specified dimension. This parameter indicates which dimension to use for grouped summation. The data type is `int32`. This is an optional input. The default value is `0`.
- `dim_size` (int): Length of the output tensor along the `dim` dimension. The data type is `int32`. This is an optional input. The default value is `None`. This parameter takes effect only when `out` is `None`.

## Returns

- `out` (Tensor): Summed tensor. The data type is `float32`.

## Constraints

- The number of dimensions of `indices` must be less than or equal to the number of dimensions of `src`, and the length of each dimension must match that of `src`.
- The values of `indices` must be valid non-negative index values. When `out` or `dim_size` is not `None`, the values of `indices` must be valid index values along the `dim` dimension of the output tensor.
- The number of dimensions of `out` must match that of `src`, and the length of every dimension except `dim` must match that of `src`.
- The value of `dim` must not exceed the number of dimensions of `indices`.
- The value of `dim_size` must be a valid non-negative length.
- `src` and `out` do not support `inf`, `-inf`, or `nan`.
- This operator performs well in both the forward and backward passes when the tail block is large. It performs poorly when the tail block is small. Here, the tail block refers to the size of the `N` dimensions after `src`, where `N = src.dim() - indices.dim()`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import scatter_add
src = torch.randn(4, 5, 6).to(torch.float)
indices = torch.randint(5, (4, 5)).to(torch.int32)
dim = 0
src.requires_grad = True
out = scatter_add(src.npu(), indices.npu(), None, dim)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
