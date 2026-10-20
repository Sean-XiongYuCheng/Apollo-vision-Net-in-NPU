# scatter_max

## Interface Prototype

```python
mx_driving.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```

## Description

Scatters the elements in the input tensor `updates` according to the indices in `indices` along dimension 0, then takes the maximum along dimension 0 and returns the maximum values and their corresponding indices. For a 1D tensor, the formulas are as follows:
$$out_i = max(out_i, max_j(updates_j))$$
$$argmax_i = argmax_j(updates_j)$$
Here, $i = indices_j$.

## Parameters

- `updates` (Tensor): Source tensor to update. The data type is `float32`.
- `indices` (Tensor): Index tensor. The data type is `int32`.
- `out` (Tensor): Tensor to be updated. The data type is `float32`. Default value: `None`.

## Returns

- `out` (Tensor): Updated tensor. The data type is `float32`.
- `argmax` (Tensor): Tensor of indices corresponding to the maximum values. The data type is `int32`.

## Constraints

- Assume that the number of elements in dimension 0 of `updates` is `N`, and the total number of elements across the remaining axes is M. Then N and M must satisfy: `N * (M + 1) < 4,026,531,840`. If this constraint is exceeded, the operator reports an out-of-memory error.
- When the axes of `updates` other than dimension 0 are combined, the result must be 32-byte aligned.
- `indices` must be 1D, and the length of dimension 0 of `indices` must match that of `updates`.
- The values in `indices` must be valid non-negative indices, and the maximum value in `indices` must be less than 491520.
- `out` must have the same dimensions as `updates`, and the lengths of its dimensions other than dimension 0 must match those of `updates`.
- Backward propagation supports only `updates` with 2 dimensions. The remaining constraints are the same as those for the forward pass.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import scatter_max
updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
updates.requires_grad = True
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
