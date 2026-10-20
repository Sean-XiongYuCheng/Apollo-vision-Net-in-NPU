# graph_softmax

## Interface Prototype

```python
mx_driving.graph_softmax(Tensor src, Tensor index) -> Tensor
```

## Description

Computes softmax over the edge feature values in `src` by grouping edges according to the nodes they point to in `index`.

## Parameters

- `src` (Tensor): Features for each edge. The data type is `float32`. The shape is `[num_edge, num_feature]`. Only `num_feature = 8` is supported.
- `index` (Tensor): Node that each edge points to. The data type is `int32`. The shape is `[num_edge]`.

## Returns

- `softmaxResult` (Tensor): Softmax result for each node group. All edges that point to the same node are treated as one group. The data type is `float32`. The shape is `[num_edge, num_feature]`.

## Constraints

- `num_feature = 8`
- `1 <= num_edge < 50000`
- `0 <= index < 1500000`
- `-500.0 <= src < 500.0`

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import mx_driving

torch.manual_seed(1)

def gen_inputs(Num_Edge, Num_Feature):
    src = torch.rand((Num_Edge, Num_Feature)) * 1000 - 500  # src range is [-500, 500)
    index = torch.randint(0, 1500000, (Num_Edge,), dtype=torch.int32)  # [0, 1500000)
    return src, index

Num_Feature = 8  # The feature number is 8 in the QCNet model
Num_Edge = 10

src, index = gen_inputs(Num_Edge, Num_Feature)
out_npu = mx_driving.graph_softmax(src.npu(), index.npu())
```
