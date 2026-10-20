# grid_sampler3d_v1

## Interface Prototype

```python
mx_driving.grid_sampler3d_v1(Tensor input, Tensor grid, str mode="bilinear", str padding_mode="zeros", bool align_corners=False) -> Tensor
```

## Description

Performs grid sampling. Given an input tensor and a corresponding grid, this operator interpolates the values at the coordinates specified by the grid and fills them into the corresponding positions in the output tensor.

## Parameters

- `input` (Tensor): Input tensor. The data type is `float32`, and the shape is $(N, C, D_{in}, H_{in}, W_{in})$.
- `grid` (Tensor): Grid tensor. The data type is `float32`, and the shape is $(N, D_{out}, H_{out}, W_{out}, 3)$. 3 represents `x`, `y`, and `z`. The values in `grid` are normalized to [-1, 1].
- `mode` (str): Interpolation mode. When the value is `"bilinear"`, the operator performs bilinear interpolation.
- `padding_mode` (str): Padding mode. It specifies how to handle out-of-bounds coordinates. When the value is `"zeros"`, the operator pads out-of-bounds positions with `0`.
- `align_corners` (bool): Mapping between feature map coordinates and feature values. The `align_corners` parameter supports only `True`. When the value is `True`, feature values are located at pixel centers.

## Returns

- `output` (Tensor): Output tensor. The data type is `float32`, and the shape is $(N, C, D_{out}, H_{out}, W_{out})$.

## Constraints

- `input` and `grid` must be 5D tensors, and they must have the same batch size.
- `input` and `grid` do not support `inf`, `-inf`, or `nan`, and they do not support empty tensors.
- The last dimension of `grid` must be `3`, and its values must be normalized to [-1, 1].
- The number of elements in `input`, `grid`, and `output` must fall within the `int32` range, and the offset of each element must be representable with 32-bit indexing.
- The `mode` parameter supports only `"bilinear"`. The `padding_mode` parameter supports only `"zeros"`. The `align_corners` parameter supports only `True`. `input` requires `C <= 128` and must be a multiple of 4.
- The operator configuration of the SparseBEV model serves as a model case. All parameter settings must match the model configuration. That is, `input: (64, 64, 6, 64, 176), grid: (64, 1460, 4, 1, 3), "bilinear", "zeros", True`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import grid_sampler3d_v1

inp = torch.rand(2, 6, 4, 4, 4, dtype=torch.float32).npu()
rand_tensor = torch.rand(2, 1, 4, 2, 3, dtype=torch.float32).npu()
grid = 2 * rand_tensor - 1
inp.requires_grad = True
grid.requires_grad = True

output = grid_sampler3d_v1(inp, grid, "bilinear", "zeros", True)
output.backward(torch.ones_like(output))
```
