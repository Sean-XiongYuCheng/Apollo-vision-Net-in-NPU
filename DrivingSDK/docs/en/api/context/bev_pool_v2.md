# bev_pool_v2

## Interface Prototype

```python
mx_driving.bev_pool_v2(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                       List[int] bev_feat_shape, Tensor interval_starts, Tensor interval_lengths) -> Tensor
```

> Note: You can replace this operator with the higher-performance [bev_pool_v3](./bev_pool_v3.md).

## Description

Optimized BEV pooling. See the paper **BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View**.

## Parameters

- `depth` (Tensor): Depth tensor. The data type is `float32`. The shape is `[B, N, D, H, W]`. `B` is the batch size, `N` is the number of feature maps, and `D`, `H`, and `W` represent depth, height, and width, respectively.
- `feat` (Tensor): Feature tensor. The data type is `float32`. The shape is `[B, N, H, W, C]`. `B` is the batch size, `N` is the number of feature maps, and `H`, `W`, and `C` represent height, width, and the number of channels, respectively.
- `ranks_depth` (Tensor): Depth rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `ranks_feat` (Tensor): Feature rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `ranks_bev` (Tensor): BEV rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `bev_feat_shape` (List[int]): BEV feature shape. The data type is `int32`. The length is 5, representing `B`, `D`, `H`, `W`, and `C`, respectively.
- `interval_starts` (Tensor): Interval start tensor. The data type is `int32`. The shape is `[N_INTERVALS]`.
- `interval_lengths` (Tensor): Interval length tensor. The data type is `int32`. The shape is `[N_INTERVALS]`.

## Returns

- `bev_pooled_feat` (Tensor): BEV pooled feature tensor. The data type is `float32`. The shape is `[B, C, D, H, W]`.

## Constraints

- The values of `ranks_depth` must fall within [0, `B*D*H*W`].
- The values of `ranks_feat` must fall within [0, `B*N*H*W`].
- The values of `ranks_bev` must fall within [0, `B*D*H*W`].
- `C <= 1024`.
- `B * D * H * W * C <= 2^31`, `B` and `D` are less than or equal to 8, and `H` and `W` are less than or equal to 256.
- `N_RANKS <= 2^21`.
- The same constraints apply to the backward pass.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import bev_pool_v2
depth = torch.rand(2, 1, 8, 256, 256).npu()
feat = torch.rand(2, 1, 256, 256, 64).npu()
feat.requires_grad_()
ranks_depth = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_feat = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_bev = torch.tensor([0, 1], dtype=torch.int32).npu()
bev_feat_shape = [2, 8, 256, 256, 64]
interval_starts = torch.tensor([0], dtype=torch.int32).npu()
interval_lengths = torch.tensor([2], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```
