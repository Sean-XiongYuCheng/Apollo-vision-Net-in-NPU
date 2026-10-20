# dynamic_scatter

## Interface Prototype

```python
mx_driving.dynamic_scatter(Tensor feats, Tensor coors, str reduce_type = 'max') -> Tuple[torch.Tensor, torch.Tensor]
```

## Description

Compress the point cloud features in their corresponding voxels.

## Parameters

- `feats` (Tensor): Point cloud feature tensor `[N, C]`. Only two dimensions are supported. The data type is `float32`. The feature vector length `C` can be up to 2048.
- `coors` (Tensor): Voxel coordinate mapping tensor `[N, 3]`. Only two dimensions are supported. The data type is `int32`. Here, `x`, `y`, and `z` represent the three voxel coordinates, and their values must satisfy `0 <= x, y < 2048` and `0 <= z < 256`.
- `reduce_type` (str): Reduction type. Optional values are `max`, `mean`, and `sum`. The default value is `max`.

## Returns

- `voxel_feats` (Tensor): Compressed voxel feature tensor. Only two dimensions are supported. The data type is `float32`.
- `voxel_coors` (Tensor): Deduplicated voxel coordinates. Only two dimensions are supported. The data type is `int32`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import dynamic_scatter

feats = torch.tensor([[1, 2, 3], [3, 2, 1], [7, 8, 9], [9, 8, 7]], dtype=torch.float32).npu()
feats.requires_grad = True
coors = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.int32).npu()
voxel_feats, voxel_coors = dynamic_scatter(feats, coors, 'max')
voxel_feats.backward(torch.ones_like(voxel_feats))

```
