# hungarian_match[beta]

## 接口原型

```python
mx_driving.hungarian_match(Tensor cost_matrix) -> Tuple[Tensor, Tensor]
```

## 功能描述

求解线性分配问题（Linear Assignment Problem），即给定一个代价矩阵，找到一组行-列匹配使得总代价最小。该算子基于匈牙利算法（Hungarian Algorithm，亦称 Kuhn-Munkres 算法）实现，等价于 `scipy.optimize.linear_sum_assignment` 的功能。

## 参数说明

- `cost_matrix(Tensor)`：代价矩阵，数据类型为`float32`。Shape为`[N, M]`，其中`N`为行数，`M`为列数。

## 返回值

- `row_ind(Tensor)`：最优分配的行索引，数据类型为`int32`，Shape为`[min(N, M)]`。
- `col_ind(Tensor)`：最优分配的列索引，数据类型为`int32`，Shape为`[min(N, M)]`。

满足 $\sum_{i} \text{cost\_matrix}[\text{row\_ind}[i], \text{col\_ind}[i]]$ 为所有可行分配中的最小值。

## 约束说明

1. 输入代价矩阵`cost_matrix`的数据类型仅支持`float32`。
2. 当前版本仅支持`N`与`M`小于等于1024。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>

## 调用示例

```python
import torch
import torch_npu
from mx_driving import hungarian_match

cost_matrix = torch.tensor([
    [10, 5, 13],
    [3, 6, 1],
    [7, 8, 4]
], dtype=torch.float32).npu()

row_ind, col_ind = hungarian_match(cost_matrix)
```
