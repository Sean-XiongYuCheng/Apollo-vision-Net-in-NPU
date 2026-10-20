# SubMConv3d

## 接口原型

```python
mx_driving.SubMConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```

## 功能描述

稀疏卷积，只有当卷积核中心参与计算时，才会影响输出。

## 参数说明

- `in_channels(int)`：输入数据的通道数。
- `out_channels(int)`：输出通道数。
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小。
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长。
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小。
- `groups(int)`：分组卷积。
- `bias(bool)`：偏置项。
- `indice_key(str)`：该输入用于复用之前计算的索引信息。
- `mode(str)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积。

## 返回值

- `SparseConvTensor(Tensor)`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的`spatital_shape`。`batch_size` 为SparseConvTensor的batch大小；`spatial_shape`为体素网格的三维空间尺寸 `[D, H, W]`，其中 `D`、`H`、`W` 分别为深度、高度、宽度三个方向上的体素数量。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>

## 约束说明

- `kernel_size`当前支持数据类型为三维List/Tuple或Int，当前值仅支持1、3。
- `stride`当前支持数据类型为三维List/Tuple或Int，当前仅支持值为1。
- `dilation`，`groups`当前仅支持值为1。
- 对于反向也有同样的约束。
- SparseConvTensor的大小应满足：当`spatial_shape`（`D` × `H` × `W`） <= `4 * 10^8`时，`batch_size` × `D` × `H` × `W` < `2^34`；当`spatial_shape`（`D` × `H` × `W`） > `4 * 10^8`时，`batch_size` × `H` × `W` < `2^31`。
- HF32开关说明：Ascend 950系列产品可通过`torch.npu.conv.allow_hf32`控制SubMConv3d正向和反向计算是否允许使用HF32。设置为`True`时允许使用HF32，设置为`False`时关闭 HF32。Atlas A2/A3不支持通过该参数使能HF32中间计算格式。

## 调用示例

```python
import torch,torch_npu
import numpy as np
from mx_driving import SubMConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices % (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SubMConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```
