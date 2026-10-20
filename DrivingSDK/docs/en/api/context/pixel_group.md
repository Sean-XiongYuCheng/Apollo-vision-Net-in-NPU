# pixel_group

## Interface Prototype

```python
mx_driving.pixel_group(Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label, Tensor kernel_contour, int kernel_region_num, float distance_threshold) -> List[List]
```

## Description

Groups unassigned pixels based on the embedding vectors and distances between pixels.

## Parameters

- `score` (Tensor): Foreground score matrix. The data type is `float32`, and the shape is `[Height, Width]`.
- `mask` (Tensor): Foreground mask matrix that marks pixels that can be grouped. The data type is `bool`, and the shape is `[Height, Width]`.
- `embedding` (Tensor): Feature vector. The data type is `float32`, and the shape is `[Height, Width, Embedding_dim]`.
- `kernel_label` (Tensor): Instance label of the pixels. The data type is `int32`, and the shape is `[Height, Width]`.
- `kernel_contour` (Tensor): Boundary pixels of the kernel. The data type is `uint8`, and the shape is `[Height, Width]`.
- `kernel_region_num` (int): Number of kernels, that is, groups. The data type is `int`.
- `distance_threshold` (float): Distance threshold of the embedding vectors. The data type is `float`.

## Returns

- `pixel_assignment` (List): Pixel grouping information. The data type is `float32`, and the length equals the input `kernel_region_num`.

## Constraints

- `mask = score > 0.5`.
- The value of `score` is in the range [0, 1].
- The maximum value of `kernel_label` is `kernel_region_num - 1`.
- The value of `kernel_contour` is either `0` or `1`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
import numpy as np
from mx_driving import pixel_group
H, W, dim, num = 10, 10, 8, 3
score = np.random.uniform(0, 1, [H, W]).astype(np.float32)
score = torch.from_numpy(score).npu()
mask = (score) > 0.5
embedding = np.random.uniform(0, 10, [H, W, dim]).astype(np.float32)
embedding = torch.from_numpy(embedding).npu()
kernel_label = np.random.uniform(0, num, [H, W]).astype(np.int32)
kernel_label = torch.from_numpy(kernel_label).npu()
kernel_contour = np.random.uniform(0, 1, [H, W]).astype(np.uint8)
kernel_contour = torch.from_numpy(kernel_contour).npu()
kernel_region_num = num
distance_threshold = 0.8

output = pixel_group(score, mask, embedding, kernel_label, kernel_contour, kernel_region_num, distance_threshold)
```
