# deformable_aggregation

## Interface Prototype

```python
mx_driving.deformable_aggregation(Tensor feature_maps, Tensor spatial_shape, Tensor scale_start_index, Tensor sample_locations, Tensor weight) -> Tensor
```

## Description

Deformable aggregation sparsely samples multi-timestamp, multi-view, and multi-scale features at each keypoint for each anchor instance. It then hierarchically fuses them into instance features to enable precise anchor refinement.

## Parameters

- `feature_maps` (Tensor): Feature tensor. The data type is `float32`. The shape is `[bs, num_feat, c]`, where `bs` is the batch size, `num_feat` is the size of the feature map, and `c` is the feature dimension.
- `spatial_shape` (Tensor): Feature-map shape. The data type is `int32`. The shape is `[cam, scale, 2]`, where `cam` is the number of cameras, `scale` is the number of feature maps per camera, and 2 represents `H` and `W`.
- `scale_start_index` (Tensor): Tensor for the offset position of each feature map. The data type is `int32`. The shape is `[cam, scale]`, where `cam` is the number of cameras and `scale` is the number of feature maps per camera.
- `sample_locations` (Tensor): Position tensor. The data type is `float32`. The shape is `[bs, anchor, pts, cam, 2]`, where `bs` is the batch size, `anchor` is the number of anchors, `pts` is the number of sample points, `cam` is the number of cameras, and 2 represents `y` and `x`.
- `weight` (Tensor): Weight tensor. The data type is `float32`. The shape is `[bs, anchor, pts, cam, scale, group]`, where `bs` is the batch size, `anchor` is the number of anchors, `pts` is the number of sample points, `cam` is the number of cameras, `scale` is the number of feature maps per camera, and `group` is the number of groups.

## Returns

- `output` (Tensor): Output tensor. The data type is `float32`. The shape is `[bs, anchor, c]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- `bs <= 128`.
- `num_feat` is the sum of the feature counts for all feature maps in `spatial_shape`.
- `c <= 256`, and `c / group` is an integer multiple of 8.
- `cam <= 6`.
- `scale <= 4`.
- `anchor <= 2048`.
- `pts <= 2048`.
- `group = 8`.
- The values in `sample_locations` must fall within [0, 1].
- Each input tensor contains no more than 150 million elements.
- The same constraints apply to the backward pass.

## Example

```python
import torch, torch_npu
from mx_driving import deformable_aggregation

bs, num_feat, c, cam, anchor, pts, scale, group = 1, 2816, 256, 1, 10, 2000, 1, 8

feature_maps = torch.ones_like(torch.randn(bs,num_feat ,c))
spatial_shape = torch.tensor([[[32, 88]]], dtype=torch.int32)
scale_start_index = torch.tensor([[0]], dtype=torch.int32)
sampling_location = torch.rand(bs, anchor, pts, cam, 2)
weights = torch.randn(bs, anchor, pts, cam, scale, group)
feature_maps.requires_grad = True
out = deformable_aggregation(feature_maps.npu(), spatial_shape.npu(), scale_start_index.npu(), sampling_location.npu(), weights.npu())
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
