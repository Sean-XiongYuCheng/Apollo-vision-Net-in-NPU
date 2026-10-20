# hypot

## Interface Prototype

```python
mx_driving.hypot(Tensor input, Tensor other) -> Tensor
```

## Description

Given the two legs of a right triangle, returns the hypotenuse.

## Parameters

- `input` (Tensor): Input tensor representing the first leg of a right triangle. The data type is `float32`.
- `other` (Tensor): Input tensor representing the second leg of a right triangle. The data type is `float32`.

## Returns

- `output` (Tensor): Hypotenuse of the right triangle. The data type is `float32`.

## Constraints

1. `input` and `other` must have broadcastable shapes.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import hypot
input = torch.tensor([3,3,3], dtype=torch.float32).npu()
other = torch.tensor([4,4,4], dtype=torch.float32).npu()
out = hypot(input, other) # tensor([5.,5.,5.])
```
