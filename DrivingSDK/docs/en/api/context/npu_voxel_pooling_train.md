# npu_voxel_pooling_train

## Interface Prototype

```python
mx_driving.npu_voxel_pooling_train(Tensor geom_xyz, Tensor input_features, List[int] voxel_num) -> Tensor
```

## Description

Voxelizes point cloud data.

## Parameters

- `geom_xyz` (Tensor): Voxel coordinates. The data type is `int32`. The shape is `(B, N, 3)`, where 3 represents `x`, `y`, and `z`.
- `input_features` (Tensor): Point cloud data. The data type is `float32|float16`. The shape is `(B, N, C)`.
- `voxel_num` (List[int]): Voxel grid dimensions. The data type is `int32`. The shape is `(3)`, where 3 represents the length, width, and height of the voxel grid.

## Returns

- `output` (Tensor): Output result. The data type is `float32|float16`. The shape is `[B, num_voxel_y, num_voxel_x, C]`.

## Constraints

- `B <= 128`
- `N <= 100000`
- `C <= 256`
- `num_voxel_x <= 1000`
- `num_voxel_y <= 1000`
- `num_voxel_z <= 10`
- `B * num_voxel_y * num_voxel_x * C <= 100000000`
- `B * N * C <= 100000000`
- The same constraints apply to the backward pass.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
from mx_driving import npu_voxel_pooling_train

def gen_data(geom_shape, feature_shape, coeff, batch_size, num_channels, dtype):
       geom_xyz = torch.rand(geom_shape) * coeff
       geom_xyz = geom_xyz.reshape(batch_size, -1, 3)
       geom_xyz[:, :, 2] /= 100
       geom_xyz_cpu = geom_xyz.int()
       features = torch.rand(feature_shape, dtype=dtype) - 0.5
       features_cpu = features.reshape(batch_size, -1, num_channels)

       return geom_xyz_cpu, features_cpu

dtype = torch.float32
coeff = 90
voxel_num = [128, 128, 1]
batch_size = 2
num_points = 40
num_channel = 80
xyz = 3

geom_shape = [batch_size, num_points, xyz]
feature_shape = [batch_size, num_points, num_channel]

geom_cpu, feature_cpu = gen_data(geom_shape, feature_shape, coeff, batch_size, num_channel, dtype)

geom_npu = geom_cpu.npu()
feature_npu = feature_cpu.npu()
feature_npu.requires_grad = True

result_npu = npu_voxel_pooling_train(geom_npu, feature_npu, voxel_num)
result_npu.backward(torch.ones_like(result_npu))
