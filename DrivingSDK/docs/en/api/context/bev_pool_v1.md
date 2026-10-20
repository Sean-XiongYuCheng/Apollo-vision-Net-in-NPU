# bev_pool

## Interface Prototype

```python
mx_driving.bev_pool(Tensor feat, Tensor geom_feat, int B, int D, int H, int W) -> Tensor
```

> Note: You can replace this operator with the higher-performance [bev_pool_v3](./bev_pool_v3.md).

## Description

BEV pooling. See the paper **BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation**.

## Parameters

- `feat` (Tensor): Feature tensor. The data type is `float32`. The shape is `[N, C]`, where `N` is the number of elements after the original feature tensor is flattened and `C` is the feature dimension.
- `geom_feat` (Tensor): Output coordinate tensor. The data type is `int32`. The shape is `[N, 4]`, where the four values represent `h`, `w`, `b`, and `d`.
- `B` (int): Batch size.
- `D` (int): Output pooling depth.
- `H` (int): Output pooling height.
- `W` (int): Output pooling width.

## Returns

- `bev_pooled_feat` (Tensor): BEV pooled feature tensor. The data type is `float32`. The shape is `[B, C, D, H, W]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Constraints

- The four corresponding values in `geom_feat` must fall within [0, `H`-1], [0, `W`-1], [0, `B`-1], and [0, `D`-1].
- `geom_feat` and `feat` must have the same length along dimension 0.
- `C <= 1024`.
- `B * D * H * W * C <= 2^31`, `B`, `D <= 8`, `H`, `W <= 256`.
- The same constraints apply to the backward pass.

## Example

```python
import torch, torch_npu
from mx_driving import bev_pool
feat = torch.rand(4, 256).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool(feat, geom_feat, 4, 1, 256, 256)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```
