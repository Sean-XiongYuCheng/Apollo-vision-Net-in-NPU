# npu_unique[beta]

## Interface Prototype

```python
mx_driving.npu_unique(Tensor input) -> Tensor
```

## Description

Sorts the input tensor in ascending order and removes duplicate elements. It sorts the input tensor and removes duplicates.

## Parameters

- `input` (Tensor): Input tensor. The data types supported are `float16`, `bfloat16`, `int16`, `float32`, `int32`, and `int64`. The shape can have 1 to 8 dimensions.

## Returns

- `output` (Tensor): Output tensor. The data types supported are `float16`, `bfloat16`, `int16`, `float32`, `int32`, and `int64`, and they match those of the input tensor `input`. The shape is 1D.

## Constraints

- For `int32` and `int64` inputs, each element must be in the range [-16777216, 16777216] (±2^24). Otherwise, precision loss occurs.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Example

```python
import torch, torch_npu
from mx_driving import npu_unique

rand_tensor = torch.rand(559794, dtype=torch.int64)
output = npu_unique(rand_tensor.npu())
```
