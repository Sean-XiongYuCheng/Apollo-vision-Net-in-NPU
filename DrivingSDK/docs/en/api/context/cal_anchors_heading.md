# cal_anchors_heading

## Interface Prototype

```python
mx_driving.cal_anchors_heading(Tensor anchors, Tensor origin_pos=None) -> Tensor
```

## Description

Compute the heading from the input anchors and the starting point coordinates.

## Parameters

- `anchors` (Tensor): Sequence coordinates for each anchor trajectory. The data type is `float32`. The shape is `[batch_size, anchors_num, seq_length, 2]`.
- `origin_pos` (Tensor): Optional. The starting position coordinates for each anchor. The data type is `float32`. The shape is `[batch_size, 2]`.

## Returns

- `heading` (Tensor): Heading of each point on each anchor trajectory, in radians. The data type is `float32`. The shape is `[batch_size, anchors_num, seq_length]`.

## Constraints

- `1 <= batch_size <= 2048`
- `1 <= anchors_num <= 10240`
- `1 <= seq_length <= 256`

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import mx_driving
batch_size = 2
anchors_num = 64
seq_length = 24
anchors = torch.randn((batch_size, anchors_num, seq_length, 2)).npu()
origin_pos = torch.randn((batch_size, 2)).npu()
heading = mx_driving.cal_anchors_heading(anchors, origin_pos)
```
