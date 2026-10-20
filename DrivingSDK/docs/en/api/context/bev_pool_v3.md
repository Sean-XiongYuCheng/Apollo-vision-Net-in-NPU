# bev_pool_v3

## Interface Prototype

```python
mx_driving.bev_pool_v3(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                       List[int] bev_feat_shape) -> Tensor
```

## Description

BEV pooling optimization. This is the NPU-friendly variant of `bev_pool_v1` and `bev_pool_v2`, and it is the preferred choice.

## Parameters

- `depth` (Tensor): Depth tensor. The data type is `float32`. The shape is `[B, N, D, H, W]`, where `B` is the batch size, `N` is the number of features, and `D`, `H`, and `W` represent depth, height, and width.
- `feat` (Tensor): Feature tensor. The data type is `float32`. The shape is `[B, N, H, W, C]`, where `B` is the batch size, `N` is the number of features, and `H`, `W`, and `C` represent height, width, and the number of channels.
- `ranks_depth` (Tensor): Depth rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `ranks_feat` (Tensor): Feature rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `ranks_bev` (Tensor): BEV rank tensor. The data type is `int32`. The shape is `[N_RANKS]`.
- `bev_feat_shape` (List[int]): BEV feature shape. The data type is `int32`. The length is 5, and the five values represent `B`, `D`, `H`, `W`, and `C`.

## Returns

- `bev_pooled_feat` (Tensor): BEV pooled feature tensor. The data type is `float32`. The shape is `[B, C, D, H, W]`.

## Constraints

- The value of `ranks_depth` must fall within [0, `B*D*H*W`].
- The value of `ranks_feat` must fall within [0, `B*N*H*W`].
- The value of `ranks_bev` must fall within [0, `B*D*H*W`].
- `C` must be a multiple of 8, and `C` must be less than or equal to 360.
- `B * D * H * W * C` must be less than or equal to 2^31.
- The same constraints apply to the backward pass.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

`bev_pool_v1` example

```python
import torch, torch_npu
from mx_driving import bev_pool_v3
feat = torch.rand(4, 64).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_feat_shape = [4, 1, 128, 128, 64]
bev_pooled_feat = bev_pool_v3(None, feat, None, None, geom_feat, bev_feat_shape)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```

`bev_pool_v2` example

```python
import torch, torch_npu
from mx_driving import bev_pool_v3
depth = torch.rand(2, 1, 8, 256, 256).npu()
feat = torch.rand(2, 1, 256, 256, 64).npu()
feat.requires_grad_()
depth.requires_grad_()
ranks_depth = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_feat = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_bev = torch.tensor([0, 1], dtype=torch.int32).npu()
bev_feat_shape = [2, 8, 256, 256, 64]
bev_pooled_feat = bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```

## Usage Notes

`bev_pool_v3` makes the following improvements over [`bev_pool`](./bev_pool_v1.md) and [`bev_pool_v2`](./bev_pool_v2.md):

1. It avoids using `sort` to order BEV cells. Therefore, it also avoids resorting camera features and depth in `bev_pool_v2`, which is not NPU-friendly on the Ascend platform and does not bring much benefit.
2. It avoids computing `interval_starts` and `interval_lengths`. It processes the data of a single BEV cell in parallel with atomic addition. When the output space is fixed, performance improves as the number of features (`n_ranks`) increases. Conversely, when the output space is too large, with `B*D*H*W*C` on the order of 1e8, the performance gain is not obvious.

## Replacement Recommendations

You can refer to model [BEVFusion](../../../../model_examples/BEVFusion) to replace `bev_pool`, and model [BEVDet](../../../../model_examples/BEVDet) to replace `bev_pool_v2`. More specifically, for `bev_pool`, you can replace it directly based on the call example. For `bev_pool_v2`, focus on the computation before the `bev_pool_v2` call. This function is usually named `voxel_pooling_prepare_v2`. Keep the computations for `ranks_feat`, `ranks_depth`, and `ranks_bev`, and remove the subsequent sorting and the related computations for `interval_lengths` and `interval_starts`.
