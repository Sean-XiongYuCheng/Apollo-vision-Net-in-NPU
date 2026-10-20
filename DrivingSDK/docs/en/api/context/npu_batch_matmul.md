# npu_batch_matmul

## Interface Prototype

```python
mx_driving.npu_batch_matmul(Tensor projection_mat, Tensor pts_extend) -> Tensor
```

## Description

Performs batched matrix multiplication. It is equivalent to `torch.batch_matmul`.

## Parameters

- `projection_mat` (Tensor): Projection matrix. The data type is `float32`. The shape has 4 to 6 dimensions. The last two dimensions must be `4, 4` or `3, 3`, and it must be broadcast-compatible with `pts_extend`.
- `pts_extend` (Tensor): Features of all points. The data type is `float32`. The shape has 4 to 6 dimensions. The last two dimensions must be `4, 1` or `3, 1`, and it must be broadcast-compatible with `projection_mat`.

## Returns

- `output` (Tensor): Matrix multiplication result. The data type is `float32`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

The input is 6D.

```python
import numpy as np
import torch, torch_npu
import mx_driving
projection_mat =torch.randn((6, 6, 4, 4)).npu()
pts_extend =torch.randn(6, 1220, 13, 4).npu()
projection_mat_fused = projection_mat[:, :, None, None].contiguous()
pts_extend2_fused = pts_extend[:, None, ..., None].contiguous()
projection_mat_fused.requires_grad=True
pts_extend2_fused.requires_grad=True
result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused)
grad = torch.ones_like(result)
result.backward(grad)
```

The input is 4D.

```python
import numpy as np
import torch, torch_npu
import mx_driving
projection_mat =torch.randn((6, 1220, 4, 4)).npu()
pts_extend = torch.randn(6, 1220, 4, 1).npu()
projection_mat.requires_grad=True
pts_extend.requires_grad=True
result = mx_driving.npu_batch_matmul(projection_mat, pts_extend)
grad = torch.ones_like(result)
result.backward(grad)
```
