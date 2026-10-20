# border_align

## Interface Prototype

```python
mx_driving.border_align(Tensor feature_map, Tensor rois, int pooled_size) -> Tensor
```

## Description

Extracts edge features from the input Region of Interest (RoI) boxes.

## Parameters

- `feature_map` (Tensor): Input feature map. The data type is `float32`. The shape is `[Batch_size, Channels, Height, Width]`.
- `rois` (Tensor): Coordinates of the input RoI boxes. The data type is `int32`. The shape is `[Batch_size, Height * Width, 4]`.
- `pooled_size` (int): Number of sampling points on each edge. The data type is `int`.

## Returns

- `out_features` (Tensor): Extracted RoI box features. The data type is `float32`. The shape is `[Batch_size, Channels / 4, Height * Width, 4]`.

## Constraints

- `Batch_size <= 128`.
- `Channels <= 8192`, `Channels % 4 == 0`.
- `Height <= 256`, `Width <= 256`.
- `2 <= pooled_size <= 20`.
- The same constraints apply to the backward pass.
- The operator performs better when `Channels` is larger.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import numpy as np
from mx_driving import border_align

def generate_grad_outputs(output_shape):
    grad_outputs = torch.rand(output_shape)
    return grad_outputs

def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features

def generate_rois(inputs):
    num_boxes = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
    xyxy = torch.rand(num_boxes, 4)
    xyxy[:, 0::2] = xyxy[:, 0::2] * inputs.size(3)
    xyxy[:, 1::2] = xyxy[:, 1::2] * inputs.size(2)
    xyxy[:, 2:] = xyxy[:, 0:2] + xyxy[:, 2:]
    rois = xyxy.view(inputs.shape[0], -1, 4).contiguous()
    return rois

batch_size = 2
input_channels = 16
input_height = 8
input_width = 8
pooled_size = 3
features = generate_features([batch_size, input_channels, input_height, input_width])
features.requires_grad = True
grad_output = generate_grad_outputs([batch_size, input_channels // 4, input_height * input_width, 4])
rois = generate_rois(features)
output = border_align(features.npu(), rois.npu(), pooled_size)
output.backward(grad_output.npu())
```
