# three_interpolate

## Interface Prototype

```python
mx_driving.three_interpolate(features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```

## Description

Performs weighted nearest-neighbor linear interpolation on 3D data.

## Parameters

- `features` (Tensor): Features to interpolate. The data type is `float32` or `float16`. The shape is `[B, C, M]`.
- `indices` (Tensor): Indices used to compute the target features. The data type is `int32`. The shape is `[B, N, 3]`.
  - The elements of `indices` must be less than the third dimension of `features`, that is, they must be in [0, `M`).
- `weight` (Tensor): Weights used to compute the target features. The data type is `float32` or `float16`. The shape is `[B, N, 3]`.
  - The data type of `weight` must match that of `features`.
- The size of each dimension of `features`, `indices`, and `weight` must be less than 10000.
- The size of `features`, `indices`, and `weight` must not exceed 2^24.

## Returns

- `output` (Tensor): Target feature tensor. The data type is `float32` or `float16`. The shape is `[B, C, N]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Example

```python
import torch, torch_npu
from mx_driving import three_interpolate


features = torch.tensor(
            [[[2.4350, 4.7516, 4.4995, 2.4350, 2.4350, 2.4350],
            [3.1236, 2.6278, 3.0447, 3.1236, 3.1236, 3.1236],
            [2.6732, 2.8677, 2.6436, 2.6732, 2.6732, 2.6732],
            [0.0124, 7.0150, 7.0199, 0.0124, 0.0124, 0.0124],
            [0.3207, 0.0000, 0.3411, 0.3207, 0.3207, 0.3207]],
            [[0.0000, 0.9544, 2.4532, 0.0000, 0.0000, 0.0000],
            [0.5346, 1.9176, 1.4715, 0.5346, 0.5346, 0.5346],
            [0.0000, 0.2744, 2.0842, 0.0000, 0.0000, 0.0000],
            [0.3414, 1.5063, 1.6209, 0.3414, 0.3414, 0.3414],
            [0.5814, 0.0103, 0.0000, 0.5814, 0.5814, 0.5814]]],
            ).npu()
idx = torch.tensor(
            [[[0, 1, 2], [2, 3, 4], [2, 3, 4], [0, 1, 2], [0, 1, 2], [0, 1, 3]],
            [[0, 2, 3], [1, 3, 4], [2, 1, 4], [0, 2, 4], [0, 2, 4], [0, 1, 2]]],
            ).int().npu()
weight = torch.tensor(
            [[[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 5.8155e-08, 2.2373e-08],
              [1.0000e+00, 1.7737e-08, 1.7356e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]],
             [[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 1.3651e-08, 7.7312e-09],
              [1.0000e+00, 1.7148e-08, 1.4070e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]]],
            ).npu()

features.requires_grad = True
output = three_interpolate(features, idx, weight)
grad_out_tensor = torch.ones_like(output)
output.backward(grad_out_tensor)
```
