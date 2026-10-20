# npu_fused_bias_leaky_relu

## Interface Prototype

```python
mx_driving.npu_fused_bias_leaky_relu(Tensor x, Tensor bias, float negative_slope, float scale) -> Tensor
```

## Description

Applies LeakyReLU to the biased input, that is, `x + bias`.
$$
f(x)=
\begin{cases}
scale * (x + bias), \quad x + bias \geq 0 \\
scale * negative\_slope * (x + bias), \quad x + bias < 0 \\
\end{cases}
$$

## Parameters

- `x` (Tensor): Input tensor. The data type is `float32`.
- `bias` (Tensor): 1D tensor. Its length must match the second dimension of `x`. The data type is `float32`.
- `negative_slope` (float): Scalar. The data type is `float32`.
- `scale` (float): Scalar. The data type is `float32`.

## Returns

- `output` (Tensor): Output tensor. The data type is `float32`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- 3 <= `len(x.shape)` <= 8.
- `bias` is a 1D tensor, and `bias.shape[0] == x.shape[1]`.
- The total data volume must not exceed 1 billion elements.
- `negative_slope` and `scale` must not be `inf` or `-inf`.

## Example

```python
import torch, torch_npu
import numpy as np
from mx_driving import npu_fused_bias_leaky_relu

negative_slope = -0.1
scale = 0.25

B, N, H, W = [18, 256, 232, 400]
x = np.random.uniform(1, 1, [B, N, H, W]).astype(np.float32)
x = torch.from_numpy(x)
bias = np.random.uniform(-2.0, 2.0, N).astype(np.float32)
bias = torch.from_numpy(bias)

out = npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slope, scale)
```
