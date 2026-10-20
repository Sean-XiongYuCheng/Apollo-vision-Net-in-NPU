# cartesian_to_frenet

## Interface Prototype

```python
mx_driving.cartesian_to_frenet(Tensor pt, Tensor poly_line) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

## Description

Convert coordinates from the input Cartesian coordinate system to the Frenet coordinate system based on the given `poly_line`.

## Parameters

- `pt` (Tensor): Coordinates of the point to convert in the Cartesian coordinate system. The data type is `float32`. The shape is `[batch_size, num_point, 2]`.
- `poly_line` (Tensor): Points that define the centerline. The data type is `float32`. The shape is `[batch_size, num_polyline_point, 2]`.

## Returns

- `poly_start` (Tensor): Start point on `poly_line` that corresponds to each point. The data type is `float32`. The shape is `[batch_size, num_point, 2]`.
- `poly_end` (Tensor): End point on `poly_line` that corresponds to each point. The data type is `float32`. The shape is `[batch_size, num_point, 2]`.
- `sl` (Tensor): Frenet coordinates of each point. The data type is `float32`. The shape is `[batch_size, num_point, 2]`.

## Constraints

- `1 <= batch_size <= 10000`
- `1 <= num_point <= 256`
- `1 < num_polyline_point <= 256`

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import mx_driving
batch_size = 5000
num_point = 40
num_polyline_point = 20
pt = torch.randn((batch_size, num_point, 2)).npu()
poly_line = torch.randn((batch_size, num_polyline_point, 2)).npu()
poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)
```
