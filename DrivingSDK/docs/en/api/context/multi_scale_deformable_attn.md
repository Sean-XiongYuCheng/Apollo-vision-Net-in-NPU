# multi_scale_deformable_attn(MultiScaleDeformableAttnFunction.Apply)

## Interface Prototype

```python
mx_driving.multi_scale_deformable_attn(Tensor value, Tensor value_spatial_shapes, Tensor value_level_start_index, Tensor sampling_locations, Tensor attention_weights) -> Tensor
```

## Description

The multi-scale deformable attention mechanism fuses feature maps from multiple views.

## Parameters

- `value` (Tensor): Feature tensor. The data type is `float32` or `float16`. The shape is `[bs, num_keys, num_heads, embed_dims]`. `bs` is the batch size, `num_keys` is the size of the feature map, `num_heads` is the number of heads, and `embed_dims` is the dimension of the feature map. `embed_dims` must be a multiple of 8.
- `value_spatial_shapes` (Tensor): Feature map shapes. The data type is `int32` or `int64`. The shape is `[num_levels, 2]`. `num_levels` is the number of feature maps, and 2 represents `H` and `W`.
- `value_level_start_index` (Tensor): Offset tensor. The data type is `int32` or `int64`. The shape is `[num_levels]`.
- `sampling_locations` (Tensor): Location tensor. The data type is `float32` or `float16`. The shape is `[bs, num_queries, num_heads, num_levels, num_points, 2]`. `bs` is the batch size, `num_queries` is the number of queries, `num_heads` is the number of heads, `num_levels` is the number of feature maps, `num_points` is the number of sampling points, and 2 represents `x` and `y`.
- `attention_weights` (Tensor): Weight tensor. The data type is `float32` or `float16`. The shape is `[bs, num_queries, num_heads, num_levels, num_points]`. `bs` is the batch size, `num_queries` is the number of queries, `num_heads` is the number of heads, `num_levels` is the number of feature maps, and `num_points` is the number of sampling points.

## Returns

- `output` (Tensor): Fused feature tensor. The data type is `float32` or `float16`. The shape is `[bs, num_queries, num_heads*embed_dims]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- The current version supports only `num_points * num_levels <= 64`, `num_heads <= 8`, and `embed_dims <= 256`.

## Example

```python
import torch, torch_npu
from mx_driving import multi_scale_deformable_attn
bs, num_levels, num_heads, num_points, num_queries, embed_dims = 1, 1, 4, 8, 16, 32

shapes = torch.as_tensor([(100, 100)], dtype=torch.long)
num_keys = sum((H * W).item() for H, W in shapes)

value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
sampling_locations = torch.ones(bs, num_queries, num_heads, num_levels, num_points, 2) * 0.005
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))

value.requires_grad_()
sampling_locations.requires_grad_()
attention_weights.requires_grad_()

out = multi_scale_deformable_attn(value.npu(), shapes.npu(), level_start_index.npu(), sampling_locations.npu(), attention_weights.npu())
out.backward(torch.ones_like(out))
```
