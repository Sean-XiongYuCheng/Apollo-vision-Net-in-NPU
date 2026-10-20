# npu_fake_tensor_quant[beta]

## Interface Prototype

```python
mx_driving.npu_fake_tensor_quant(Tensor inputs, Tensor amax, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

```python
mx_driving.npu_fake_tensor_quant_inplace(Tensor inputs, Tensor amax, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

```python
mx_driving.npu_fake_tensor_quant_with_axis(Tensor inputs, Tensor amax, int axis, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

## Description

Implements fake quantization for quantization-aware training (QAT). It simulates quantizing floating-point input tensors into a low-precision representation, then dequantizes them back to floating-point values. Therefore, it simulates quantization error during training.

- `npu_fake_tensor_quant`: Performs scalar quantization on the entire tensor with a single `amax` value.
- `npu_fake_tensor_quant_inplace`: Has the same behavior as `npu_fake_tensor_quant`, but modifies the input tensor in place.
- `npu_fake_tensor_quant_with_axis`: Performs per-axis quantization along the specified `axis`, with each slice using an independent `amax` value.

The quantization formula is as follows.

```python
bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1
max_bound = bound
min_bound = -(bound + int(not narrow_range))
scale = max_bound / amax
output = clamp(round(input * scale), min_bound, max_bound) / scale
```

## Parameters

### `npu_fake_tensor_quant`/`npu_fake_tensor_quant_inplace`

- `inputs` (Tensor): Input tensor. The data type is `float32` or `float16`. The shape is unrestricted.
- `amax` (Tensor): Maximum-value scalar tensor. The data type must be the same as `inputs`. The shape is `[1]`.
- `num_bits` (int): Quantization bit width, which indicates the precision of the quantized result. The data type is `int32`. The default value is `8`. Typical values are `4` or `8`.
- `is_unsigned` (bool): Whether to use unsigned quantization. The data type is `bool`. The default value is `False`.
  - `False`: Signed quantization. The quantization range is [-`bound`, `bound`].
  - `True`: Unsigned quantization. The quantization range is [0, `bound`].
- `narrow_range` (bool): Whether to use narrow-range quantization. The data type is `bool`. The default value is `True`.
  - `True`: Narrow range. `min_bound = -bound`.
  - `False`: Wide range. `min_bound = -(bound + 1)`.

### `npu_fake_tensor_quant_with_axis`

- `inputs` (Tensor): Input tensor. The data type is `float32` or `float16`. The shape is unrestricted.
- `amax` (Tensor): Per-channel maximum-value tensor. The data type must be the same as `inputs`. The shape is `[inputs.shape[axis]]`.
- `axis` (int): Quantization axis, which indicates the dimension for per-channel quantization. The data type is `int32`.
- `num_bits` (int): Quantization bit width. The default value is `8`.
- `is_unsigned` (bool): Whether to use unsigned quantization. The default value is `False`.
- `narrow_range` (bool): Whether to use narrow-range quantization. The default value is `True`.

## Returns

- `output` (Tensor): Quantized tensor. The data type is the same as `inputs`, and the shape is the same as `inputs`.

## Constraints

### General Constraints

- `inputs` and `amax` must have the same data type. Only `float32` and `float16` are supported.
- `inputs` and `amax` must be contiguous tensors.
- `inputs` does not support `inf`, `-inf`, or `nan`.
- `num_bits` must be in the range [1, 32].

### `npu_fake_tensor_quant`/`npu_fake_tensor_quant_inplace`

- The shape of `amax` must be the scalar shape `[1]`.

### `npu_fake_tensor_quant_with_axis`

- The shape of `amax` must be `[inputs.shape[axis]]`, which means it must match the length of `inputs` along the `axis` dimension.
- `axis` must be in the range [0, `inputs.dim()` - 1].
- When `axis` is the last dimension, the operator automatically switches to a special optimization mode for better performance.

## Usage Recommendations

- **Scalar quantization**: Suitable for scenarios where the entire tensor uses a single scaling factor, such as activation quantization.
- **Per-axis quantization**: Suitable for scenarios that require independent quantization for each channel, such as weight quantization. The usual value is `axis=0`.
- **In-place version**: `npu_fake_tensor_quant_inplace` can save memory, but it modifies the input tensor.

## Performance Verification

|Scenario|Shape|PyTorch Implementation Time (μs)|Operator Implementation Time (μs)|
| --- | --- | --- | --- |
|Global quantization|[100663296]|3387.99|559.09|
|Per-channel (`axis=1`)|[32,8,512,768]|4114.19|1147.98|

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Examples

### Example 1: Scalar Quantization (Activation Quantization)

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant

# Create the input tensor
inputs = torch.randn(4, 64, 32, 32, dtype=torch.float32, device='npu')

# Compute amax, which is the maximum absolute value of the entire tensor
amax = inputs.abs().max()

# Scalar quantization, using 8-bit signed narrow-range quantization
quantized = npu_fake_tensor_quant(inputs, amax, num_bits=8, is_unsigned=False, narrow_range=True)
```

### Example 2: In-Place Quantization (Memory Saving)

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant_inplace

inputs = torch.randn(1000, 256, dtype=torch.float32, device='npu')

amax = inputs.abs().max()

# In-place quantization. The input tensor is modified
quantized = npu_fake_tensor_quant_inplace(inputs, amax, num_bits=8)
```

### Example 3: Per-Axis Quantization (Weight Quantization)

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant_with_axis

# Create a 4D weight tensor (N, C, H, W)
weight = torch.randn(64, 128, 3, 3, dtype=torch.float32, device='npu')

# Compute the per-channel amax along the channel axis (`axis=0`)
amax = weight.abs().amax(dim=[1, 2, 3])  # Shape: [64]

# Per-channel quantization, using 8-bit signed narrow-range quantization
quantized_weight = npu_fake_tensor_quant_with_axis(
    weight, amax, axis=0, num_bits=8, is_unsigned=False, narrow_range=True
)
```

## Use Cases

1. **Quantization-aware training (QAT)**: Simulates quantization error during training to improve model accuracy for INT8 inference.
2. **Model compression**: Converts FP32 models into low-precision representations to reduce memory usage and bandwidth requirements.
3. **Mixed-precision training**: Uses quantization for some layers and keeps FP32 for others to balance accuracy and performance.
4. **Activation quantization**: Uses scalar quantization (`npu_fake_tensor_quant`).
5. **Weight quantization**: Uses per-axis quantization (`npu_fake_tensor_quant_with_axis`).
