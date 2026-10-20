# SparseInverseConv3d

## Interface Prototype

```python
mx_driving.SparseInverseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```

## Description

Sparse inverse convolution.

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

## Constraints

- `in_channels` and `out_channels` (int) are divisible by 8.
- `kernel_size` currently supports 3D List, Tuple, or int, with values in [1, 3].
- `stride` currently supports 3D List, Tuple, or int.
- `dilation` and `groups` currently support only the value `1`.
- For both modes (`mmcv` and `spconv`), the weight shape is `[D, H, W, out_channels, in_channels]`.
- The same constraints apply to the backward pass.

## Example

```python
import torch, torch_npu
import numpy as np
from mx_driving import SparseConv3d, SparseConvTensor, SparseInverseConv3d


def generate_sparse_data(shape, num_points, num_channels, batch_size=4):
    dense_shape = shape
    ndim = len(dense_shape)
    batch_indices = []

    coors_total = np.stack(np.meshgrid(*[np.arange(0, s) for s in shape]), axis=-1)
    coors_total = coors_total.reshape(-1, ndim)

    for i in range(batch_size):
        np.random.shuffle(coors_total)
        inds_total = coors_total[:num_points]
        inds_total = np.pad(inds_total, ((0, 0), (0, 1)), mode="constant", constant_values=i)
        batch_indices.append(inds_total)

    sparse_data = np.random.uniform(-1, 1, size=[num_points * batch_size, num_channels]).astype(np.float32)
    batch_indices = np.concatenate(batch_indices, axis=0)

    return {"features": sparse_data, "indices": batch_indices.astype(np.int32)}


spatial_shape = [9, 9, 9]
in_channels, out_channels = 192, 256
kernel_size, stride = 3, 2
batch_size, feature_num = 4, 1529

sparse_dict = generate_sparse_data(spatial_shape, feature_num, in_channels, batch_size)
voxels = torch.from_numpy(sparse_dict["features"]).float()
coors = torch.from_numpy(sparse_dict["indices"][:, [3, 0, 1, 2]]).int()

conv3d = (
    SparseConv3d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, indice_key="test"
    )
    .npu()
    .float()
)

inv_conv3d = (
    SparseInverseConv3d(
        in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, indice_key="test"
    )
    .npu()
    .float()
)

x = SparseConvTensor(voxels.npu(), coors.npu(), spatial_shape, batch_size)
x1 = conv3d(x)
x2 = inv_conv3d(x1)

print("voxels.shape:", voxels.shape)
print("x1.features.shape:", x1.features.shape)
print("x2.features.shape:", x2.features.shape)
```
