# voxelization

## 接口原型

```python
mx_driving.Voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True, str layout="XYZ") -> Tensor
```

## 功能描述

将点云数据进行体素化。

## 参数说明

- `points(Tensor)`：点云数据，数据类型为`float32`。shape为`[N, F]`。其中N为点的数量，F分别代表每个点的特征维度，其中`N > 0, F >= 3`。
- `voxel_size(List[float])`：体素大小，数据类型为`float32`。shape为`[3]`。其中3分别代表`x, y, z`。
- `coors_range(List[float])`：体素范围，数据类型为`float32`。shape为`[6]`。其中6分别代表`x_min, y_min, z_min, x_max, y_max, z_max`。
- `max_points(int)`：每个体素的最大点数。默认值为`-1`。`max_points<=1000`。
- `max_voxels(int)`：最大体素数。默认值为`-1`。
- `deterministic(bool)`：是否确定性。默认值为`True`。
- `layout(str)`: 体素坐标的排序方式，默认值为`XYZ`。替换MMCV算子时使用`ZYX`，替换mmdetection3d算子时使用`XYZ`。

## API语义

当max_points=-1或max_voxels=-1时，进行`dynamic_voxelize`计算，否则进行`hard_voxelize`。

## 返回值

进行`dynamic_voxelize`计算时：

- `coors(Tensor)`：每个点所属的体素坐标，数据类型为`int32`。shape为`[N, 3]`。

进行`hard_voxelize`计算时：

- `voxel_num(int)`：去重后的体素数量。
- `vox_points(Tensor)`：每个体素对应的点云数据，数据类型为`float32`。shape为`[voxel_num, max_points, F]`。
- `coors(Tensor)`：去重后的体素坐标，数据类型为`int32`。shape为`[voxel_num, 3]`。
- `num_points_per_voxel(Tensor)`：每个体素的点云数量，数据类型为`int32`。shape为`[voxel_num]`。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>

## 调用示例

```python
import torch, torch_npu
from mx_driving import Voxelization
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 0, 9]
max_num_points = -1
max_voxels = 3500
vlz = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)
points = torch.rand(16, 3, dtype = torch.float32).npu()
results = vlz(points)
```
