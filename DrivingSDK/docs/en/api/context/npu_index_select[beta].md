# npu_index_select[beta]

## Interface Prototype

```python
mx_driving.npu_index_select(Tensor feature, Int dim, Tensor index) -> Tensor
```

## Description

Selects elements from the specified dimension `dim` of the input `feature` tensor according to the indices in `index`, and stores the results in `out`.
For example, given the input tensor $feature=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ and the index tensor $index=[1, 0]$,
the output of $mx\_driving.npu\_index\_select(feature, 0, index)$ is $out=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$.

## Parameters

- `feature` (Tensor): Tensor to select from. The supported data types are `FLOAT`, `FLOAT16`, `INT32`, and `INT16`. Only 2D tensors are supported. Non-contiguous tensors are supported.
- `dim` (Int): Dimension to select from. The supported data type is `INT64`.
- `index` (Tensor): Selection indices. The supported data types are `INT64` and `INT32`. Only 1D tensors are supported. Non-contiguous tensors are supported.

## Returns

- `out` (Tensor): Output tensor. The data type is the same as `feature`. It is 2D, and its shape is `[index.shape[0], feature.shape[1]]`.

## Constraints

- `feature` only supports 2D tensors.
- `dim` only supports `0` and `-2`.
- `index` does not support negative indices or out-of-bounds indices. Therefore, the valid range is [0, `feature.shape[0]`).
- This API depends on the `aclnnIndexAddV2` interface. Therefore, you need to use a CANN package released after June 18, 2025, for it to take effect.
- The backward pass has the same constraints.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>

## Example

```python
import torch, torch_npu
from mx_driving import npu_index_select

x = torch.randn(3, 4)
index = torch.tensor([0, 2])
npu_output = npu_index_select(x.npu(), 0, index.npu())
npu_output.backward(torch.ones_like(npu_output))
```
