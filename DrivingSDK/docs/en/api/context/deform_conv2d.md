# deform_conv2d(DeformConv2dFunction.apply)

## Interface Prototype

```python
mx_driving.deform_conv2d(Tensor x, Tensor offset, Tensor weight, Union[int, Tuple[int, ...]] stride, Union[int, Tuple[int, ...]] padding, Union[int, Tuple[int, ...]] dilation, int groups, int deformable_groups) -> Tensor
```

## Description

Deformable convolution.

## Parameters

- `x` (Tensor): Input features. The data type is `float32`. The shape is `(n, c_in, h_in, w_in)`, where `n` is the batch size, `c_in` is the number of input channels, `h_in` is the height of the input feature map, and `w_in` is the width of the input feature map.
- `offset` (Tensor): Offsets. The data type is `float32`. The shape is `(n, 2 * k * k, h_out, w_out)`, where `n` is the batch size, `k` is the kernel size, `h_out` is the height of the output feature map, and `w_out` is the width of the output feature map.
- `weight` (Tensor): Kernel weights. The data type is `float32`. The shape is `(c_out, c_in, k, k)`, where `c_out` is the number of output channels, `c_in` is the number of input channels, and `k` is the kernel size.
- `stride` (Union[int, Tuple[int, ...]]): Convolution stride.
- `padding` (Union[int, Tuple[int, ...]]): Convolution padding.
- `dilation` (Union[int, Tuple[int, ...]]): Dilated convolution size.
- `groups` (int): Group convolution size. Both `c_in` and `c_out` must be divisible by `groups`.
- `deformable_groups` (int): Split the channels into groups when computing offsets. Currently only 1 is supported.

## Returns

- `output` (Tensor): Output tensor. The data type is `float32`. The shape is `(n, c_out, h_out, w_out)`, where `n` is the batch size, `c_out` is the number of output channels, `h_out` is the height of the output feature map, and `w_out` is the width of the output feature map.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

1. `deformable_groups` currently supports only `1`.
2. `h_in`, `w_in`, `h_out`, and `w_out` must satisfy the following conditions:
    $$
    w_{out}=(w_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 \\
    h_{out}=(h_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1
    $$
3. `c_in` must be a multiple of 64.
4. On Ascend 950 products, you can set `torch.npu.conv.allow_hf32=True` to control whether the operator can use `hf32` computation as the intermediate computation format. On Atlas A2 training products and Atlas A3 training products, this parameter cannot be used to enable the `hf32` intermediate computation format.

## Example

```python
import torch
import torch_npu
from mx_driving import deform_conv2d, DeformConv2dFunction

n, c_in, h_in, w_in = 16, 64, 100, 200
c_out, k, h_out, w_out = 64, 3, 50, 100

x = torch.randn((n, c_in, h_in, w_in)).npu()
offset = torch.randn((n, 2 * k * k, h_out, w_out)).npu()
weight = torch.randn((c_out, c_in, k, k)).npu()
x.requires_grad_()
offset.requires_grad_()
weight.requires_grad_()
stride = 1
padding = 1
dilation = 1
groups = 1
deformable_groups = 1

output = deform_conv2d(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
output = DeformConv2dFunction.apply(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
output.backward(torch.ones_like(output))
```
