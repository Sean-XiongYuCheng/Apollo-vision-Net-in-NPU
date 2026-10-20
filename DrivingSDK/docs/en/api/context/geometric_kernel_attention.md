# geometric_kernel_attention

## Interface Prototype

```python
mx_driving.geometric_kernel_attention(Tensor value, Tensor spatial_shapes, Tensor level_start_index,
                                      Tensor sampling_locations, Tensor attention_weights) -> Tensor
```

## Description

Computes attention features for sampling points based on their weights.

## Parameters

- `value` (Tensor): Feature map tensor. The data type is `float32`. The shape is `[B, num_keys, num_heads, dim]`.
- `spatial_shapes` (Tensor): Spatial-shape tensor for the feature map. The data type is `int32`. The shape is `[num_levels, 2]`. The last dimension is fixed at 2, representing `[H, W]` for the corresponding level.
- `level_start_index` (Tensor): Start-index tensor for each level. The data type is `int`. The shape is `[num_levels, ]`.
- `sampling_locations` (Tensor): Sampling-location tensor. The data type is `float32`. The shape is `[B, num_queries, num_heads, num_levels, num_points, 2]`. The last dimension is fixed at 2, representing the coordinate pair `[w, h]`.
- `attention_weights` (Tensor): Attention-weight tensor. The data type is `float32`. The shape is `[B, num_queries, num_heads, num_levels, num_points]`.

## Returns

- `output` (Tensor): Output tensor after attention weighting. The data type is `float32`. The shape is `[B, num_queries, num_heads*dim]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- Assume that `num_points` is rounded up to the nearest multiple of 8 to obtain `num_points_align`. In the current version, the backward operator supports only `num_heads * num_levels * num_points_align * dim < 16 * 1024` and `num_heads * num_levels * num_points_align < 256`.
- In the current version, the backward operator supports only `dim % 8 == 0`.

## Example

```python
import torch, torch_npu
from mx_driving import geometric_kernel_attention

bs, num_queries, embed_dims, num_heads, num_levels, num_points = [6, 9680, 32, 8, 4, 4]
shapes = torch.tensor([60, 40] * num_levels).reshape(num_levels, 2)
num_keys = sum((H * W) for H, W in shapes)
value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01 # Initialize the value tensor with random values scaled by 0.01
sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
sampling_locations.uniform_(0, 40)
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
offset = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
grad_output = torch.rand(bs, num_queries, num_heads * embed_dims) * 1e-3

npu_value = value.float().npu()
npu_shapes = shapes.int().npu()
npu_offset = offset.int().npu()
npu_sampling_locations = sampling_locations.float().npu()
npu_attention_weights = attention_weights.float().npu()
npu_grad_output = grad_output.float().npu()

npu_value.requires_grad_()
npu_attention_weights.requires_grad_()

npu_output = geometric_kernel_attention(npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights)
npu_output.backward(npu_grad_output)
```
