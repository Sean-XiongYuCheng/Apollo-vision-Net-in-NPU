# grid_sampler2d_v2

## Interface Prototype

```python
mx_driving.grid_sampler2d_v2(Tensor input, Tensor grid, str mode="bilinear", str padding_mode="zeros", bool align_corners=False) -> Tensor
```

## Description

Performs grid sampling. Given an input tensor and a corresponding grid, it samples values from the input tensor according to the coordinates at each grid position and places them at the corresponding positions in the output tensor.

## Parameters

- `input` (Tensor): Input tensor. The data type is `float32`, and the shape is $(N, C, H_{in}, W_{in})$.
- `grid` (Tensor): Grid tensor. The data type is `float32`, and the shape is $(N, H_{out}, W_{out}, 2)$. 2 represents `x` and `y`. The grid values are usually normalized to [-1, 1].
- `mode` (str): Interpolation mode. When the value is `"bilinear"`, the operator performs bilinear interpolation.
- `padding_mode` (str): Padding mode. It specifies how to handle out-of-bounds coordinates. When the value is `"zeros"`, out-of-bounds positions are padded with `0`. When the value is `"border"`, the operator pads them with border values.
- `align_corners` (bool): Specifies how feature map coordinates correspond to feature values. When the value is `True`, feature values are located at pixel centers. When the value is `False`, feature values are located at pixel corners.

## Returns

- `output` (Tensor): Output tensor. The data type is `float32`, and the shape is $(N, C, H_{out}, W_{out})$.

## Constraints

- `input` and `grid` must be 4D tensors, and they must have the same batch size.
- `input` and `grid` do not support `inf`, `-inf`, or `nan`, and they do not support empty tensors.
- The last dimension of `grid` must be `2`, and its values must be normalized to [-1, 1].
- The number of elements in `input`, `grid`, and `output` must be within the `int32` range, and the offset of each element must be representable with 32-bit indexing.
- `mode` supports only `"bilinear"`, and `padding_mode` supports only `"zeros"` and `"border"`. `input` requires `C <= 128`, and `C` must be a multiple of 4.
- Compared with `grid_sample` in CANN, this operator is optimized for the BEVDet model scenario. All parameter settings must match the model configuration. Specifically, `input: (24, 4, 64, 176), grid: (24, 5632, 176, 2), "bilinear", "zeros", True`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import grid_sampler2d_v2

inp = torch.rand(1, 4, 8, 8, dtype=torch.float32).npu()
rand_tensor = torch.rand(1, 2, 3, 2, dtype=torch.float32).npu()
grid = 2 * rand_tensor - 1
inp.requires_grad = True
grid.requires_grad = True

output = grid_sampler2d_v2(inp, grid, "bilinear", "zeros", False)
output.backward(torch.ones_like(output))
```
