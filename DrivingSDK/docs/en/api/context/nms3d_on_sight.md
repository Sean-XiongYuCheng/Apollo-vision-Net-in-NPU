# nms3d_on_sight

## Interface Prototype

```python
mx_driving.nms3d_on_sight(Tensor boxes, Tensor scores, float threshold) -> Tensor
```

## Description

In the BEV view, this operator removes candidate boxes from the 3D box set whose BEV distance (`dist_bev`) exceeds the specified threshold (`threshold`).

## Parameters

- `boxes` (Tensor): Input 3D boxes. The data type is `float32`. Each box has the shape `[x, y, z, dx, dy, dz, heading]`, and `boxes` has the shape `[N, 7]`.
- `scores` (Tensor): Confidence scores of the input candidate boxes. The data type is `float32`. The shape is `[N]`.
- `threshold` (float): Threshold used to calculate the BEV distance. The data type is `float32`.

## Returns

- `order` (Tensor): Indices of the retained boxes.

## Constraints

- The ranges of `dx`, `dy`, and `dz` are (0, 100).
- The range of `heading` is (0, 1).
- 0 < `N` <= 2500.
- 0 <= `scores` <= 1.
- -0.5 <= `threshold` <= 1.
- When distances are equal, sorting is unstable. Therefore, the distance precision meets the requirement, but the index precision can be incorrect. As a result, the output cannot exactly match competing products.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import numpy as np
import torch
import torch_npu
from mx_driving import nms3d_on_sight

def generate_data_numpy(N):
    np.random.seed(7)
    shape = [N, 7]
    boxes_np = np.round(np.random.uniform(0, 100, shape), 2).astype(np.float32)
    angle_np = np.round(np.random.uniform(0, 1, shape[0]), 2).astype(np.float32)
    boxes_np[:, 6] = angle_np
    scores_np = np.round(np.random.uniform(0, 1, shape[0]), 4).astype(np.float32)
    boxes = torch.from_numpy(boxes_np)
    scores = torch.from_numpy(scores_np)
    threshold = np.random.uniform(-0.5, 1)
    return boxes, scores, threshold

boxes, scores, threshold = generate_data_numpy(N=100)
result = nms3d_on_sight(boxes.npu(), scores.npu(), threshold)
```
