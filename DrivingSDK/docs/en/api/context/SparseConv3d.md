# SparseConv3d

## Interface Prototype

```python
mx_driving.SparseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```

## Description

Sparse convolution.

## Parameters

- `in_channels` (int): Number of input channels.
- `out_channels` (int): Number of output channels.
- `kernel_size` (List(int)/Tuple(int)/int): Size of the convolution kernel in a convolutional neural network.
- `stride` (List(int)/Tuple(int)/int): Step size when the convolution kernel slides over the input data.
- `dilation` (List(int)/Tuple(int)/int): Dilated convolution size.
- `groups` (int): Grouped convolution.
- `bias` (bool): Bias term.
- `indice_key` (str): Used to reuse previously computed index information for this input.
- `mode` (str): Distinguishes sparse convolution in the `mmcv` and `spconv` frameworks.

## Returns

- `SparseConvTensor` (Tensor): Stores the output feature values `out_feature`, the corresponding index positions `out_indices`, and the corresponding `spatial_shape`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- `kernel_size` currently supports 3D List, Tuple, or int, with values in [1, 3].
- `stride` currently supports 3D List, Tuple, or int.
- `dilation` and `groups` currently support only the value `1`.
- The same constraints apply to the backward pass.

## Example

```python
import torch,torch_npu
import numpy as np
from mx_driving import SparseConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices % (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = tensor_uniform = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```
