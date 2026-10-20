# radius

## Interface Prototype

```python
mx_driving.radius(Tensor x,Tensor y,Tensor ptr_x, Tensor ptr_y,
                  float r, int max_num_neighbors) -> Tensor
```

## Description

Given the `Ndim`-dimensional coordinates `X` and `Y` of two point sets, for each point `y` in `Y`, this operator finds the indices of all points in `X` that belong to the same batch as `y` and lie within the radius `r`.

## Parameters

- `x` (Tensor): `Ndim`-dimensional coordinates of the first point set. The data type is `float32`, and the shape is `[numpoints_x, Ndim]`.
- `y` (Tensor): `Ndim`-dimensional coordinates of the second point set. The data type is `float32`, and the shape is `[numpoints_y, Ndim]`.
- `ptr_x` (Tensor): Batch split offsets of the first point set. The data type is `int`, and the shape is `[batch_size + 1]`. The value of `ptr_x[0]` is `0`, and the subsequent values strictly increase. The value of `ptr_x[batch_size]` is `numpoints_x`. `X[ptr_x[0]: ptr_x[1]]` belongs to the first batch, `X[ptr_x[1]: ptr_x[2]]` belongs to the second batch, and the partition continues in the same way for the remaining points.
- `ptr_y` (Tensor): Batch split offsets of the second point set. The data type is `int`, and the shape is `[batch_size + 1]`. The value of `ptr_y[0]` is `0`, and the subsequent values strictly increase. The value of `ptr_y[batch_size]` is `numpoints_y`. `Y[ptr_y[0]: ptr_y[1]]` belongs to the first batch, `Y[ptr_y[1]: ptr_y[2]]` belongs to the second batch, and the partition continues in the same way for the remaining points.
- `r` (float): Radius. The data type is `float`.
- `max_num_neighbors` (int): Maximum number of neighbors. The data type is `int`. For any point `y`, if the number of `x` points within radius `r` is greater than `max_num_neighbors`, the operator returns only the indices of the first `max_num_neighbors` `x` points in index order.

## Returns

- `output_index` (Tensor): All `y`-`x` neighbor index pairs that meet the requirements. The data type is `int`, and the shape is `[2, num_neighbors]`. `num_neighbors` indicates the total number of neighbors. You can obtain its value only after the operator finishes computing.

## Constraints

- `Ndim < 9`
- `batch_size <= 1024`
- The number of points in each batch, that is, the interval between adjacent values in `ptr_x` and `ptr_y`, is less than or equal to 512.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch
import torch_npu
import numpy as np
from mx_driving import radius

def gen_points(num_points, data_range, Ndim):
    points = 2 * data_range * (torch.rand([num_points, Ndim]) - 0.5)
    return points

def gen_batch_ptr(batch_size, max_points_per_batch):
    batch_list = torch.randint(0, max_points_per_batch, [batch_size]).int()
    batch_ptr = torch.cumsum(batch_list, dim = 0).int()
    batch_ptr = torch.cat([torch.zeros([1]).int(), batch_ptr])
    return batch_ptr

def gen_inputs(data_range, batch_size, max_points_per_batch, Ndim):
    ptr_x = gen_batch_ptr(batch_size, max_points_per_batch)
    ptr_y = gen_batch_ptr(batch_size, max_points_per_batch)
    num_points_x = ptr_x[-1]
    num_points_y = ptr_y[-1]
    x = gen_points(num_points_x, data_range, Ndim)
    y = gen_points(num_points_y, data_range, Ndim)
    return x, y, ptr_x, ptr_y

data_range = 50 # X和Y的取值在[-50, 50]范围内
batch_size = 16
max_points_per_batch = 512 # ptr_x和ptr_y相邻点间隔不大于512
r = 20.0
max_num_neighbors = 100
Ndim = 2 # 最大支持8维

x, y, ptr_x, ptr_y = gen_inputs(data_range, batch_size, max_points_per_batch, Ndim)
out_npu = radius(x.npu(), y.npu(), ptr_x.npu(), ptr_y.npu(), r, max_num_neighbors)
```
