# npu_add_relu

## Interface Prototype

```python
mx_driving.npu_add_relu(Tensor x, Tensor y) -> Tensor
```

## Description

Performs the same function as `relu(x + y)`.

## Parameters

- `x` (Tensor): Input data. The data type is `float32`, and the shape is unrestricted.
- `y` (Tensor): Input data. The data type is `float32`, and the shape must match `x`.

## Returns

- `output` (Tensor): Output data. The data type is `float32`, and the shape matches `x`.

## Constraints

- The product of all dimensions in the `x` or `y` shape must stay within the valid representable range of the `int32` data type. Otherwise, a computation error occurs.
- Input `x` and input `y` must have the same shape and `dtype`. Broadcasting is not supported.
- This operator provides a performance gain over `relu(x + y)` only when `x` contains more than 2,000,000 elements.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving.fused import npu_add_relu
x = torch.tensor([[[1, 2, 3], [-1, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
y = torch.tensor([[[1, 2, 3], [-1, -2, 6], [7, 8, 9]]], dtype=torch.float32).npu()
x.requires_grad = True
y.requires_grad = True
out = npu_add_relu(x, y)
out.backward(torch.ones_like(out))
```
