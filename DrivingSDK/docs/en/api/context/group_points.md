# group_points

## Interface Prototype

```python
mx_driving.group_points(Tensor features, Tensor indices) -> Tensor
```

## Description

Groups point cloud data based on the indices.

## Parameters

- `features` (Tensor): Features to be interpolated. The data type is `float32`. The shape is `[B, C, N]`.
- `indices` (Tensor): Indices used to compute the target features. The data type is `int32`. The shape is `[B, npoints, nsample]`.

## Returns

- `output` (Tensor): Grouped point cloud data. The data type is `float32`. The shape is `[B, C, npoints, nsample]`.

## Constraints

- `features` and `indices` have the same first dimension, and neither supports scalars or empty tensors.
- The values of `indices` must be less than the third dimension of `features`, that is, in [0, `N`).
- `C <= 1024`.
- The backward pass has the same constraints.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
from mx_driving import group_points


indices = torch.tensor([[[0, 2, 5, 5], [1, 0, 5, 0], [2, 1, 4, 4]]]).int().npu()
features = torch.tensor([[[0.9178, -0.7250, -1.6587, 0.0715, -0.2252, 0.4994],
                          [0.6190, 0.1755, -1.7902, -0.5852, -0.3311, 1.9764],
                          [1.7567, 0.0740, -1.1414, 0.4705, -0.3197, 1.1944],
                          [-0.2343, 0.1194, 0.4306, 1.3780, -1.4282, -0.6377],
                          [0.7239, 0.2321, -0.6578, -1.1395, -2.3874, 1.1281]]],
                          dtype=torch.float32).npu()
features.requires_grad = True
output = group_points(features, indices)
grad_out_tensor = torch.ones_like(output)
output.backward(grad_out_tensor)
```
