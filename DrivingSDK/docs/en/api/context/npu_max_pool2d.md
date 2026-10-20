# npu_max_pool2d

## Interface Prototype

```python
mx_driving.npu_max_pool2d(Tensor x, int kernel_size, int stride, int padding) -> Tensor
```

## Description

Performs max pooling on the input and returns the pooled maximum value.

## Parameters

- `x` (Tensor): Input tensor to pool. The data type is `float32`. The format is NCHW. The amount of input data does not exceed 1 billion.
- `kernel_size` (int): Kernel size. Currently, only 3 is supported.
- `stride` (int): Stride of the pooling kernel as it slides over the input data. Currently, only 2 is supported.
- `padding` (int): Padding. Currently, only 1 is supported.

## Returns

- `y` (Tensor): Maximum pooled value. The data type is `float32`. The format is NCHW.

## Constraints

`kernel_size` supports only `3`, `stride` supports only `2`, and `padding` supports only `1`. The amount of data along the `C` axis in the input must be a multiple of 8, and `H` and `W` must be greater than 100.
`C <= 1024`. Performance is better for larger `C` values. You are advised to use `C >= 64`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import npu_max_pool2d
kernel_size = 3
stride = 2
padding = 1
x = torch.randn(18, 64, 464, 800).npu()
res = npu_max_pool2d(x, kernel_size, stride, padding)
```
