# unique_voxel

## Interface Prototype

```python
mx_driving.unique_voxel(Tensor voxels) -> int, Tensor, Tensor, Tensor, Tensor
```

## Description

Deduplicates the input point cloud data.

## Parameters

- `voxels` (Tensor): Input indices. The data type is `int32`. The shape is `[N]`.

## Returns

- `num_voxels` (int): Number of voxels.
- `uni_voxels` (Tensor): Deduplicated voxel data. The data type is `int32`. The shape is `[num_voxels]`.
- `uni_indices` (Tensor): Deduplicated index data. The data type is `int32`. The shape is `[num_voxels]`.
- `argsort_indices` (Tensor): Sorted index data. The data type is `int32`. The shape is `[N]`.
- `uni_argsort_indices` (Tensor): Deduplicated and sorted index data. The data type is `int32`. The shape is `[num_voxels]`.

## Constraints

The size of `N` is limited by available memory, and you are advised to keep `N` less than or equal to 2^32.

Due to chip instruction limitations, the input data type can only be `int32`, and values must be greater than or equal to 0 and less than 2^30.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import numpy as np
from mx_driving import unique_voxel
voxels = np.random.randint(0, 1024, (100000,)).astype(np.int32)
voxels_npu = torch.from_numpy(voxels).npu()
num_voxels, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices = unique_voxel(voxels_npu)

```
