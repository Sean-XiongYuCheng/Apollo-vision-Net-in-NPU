# npu_fft_conv1d[beta]

## 接口原型

```python
mx_driving.npu_fft_conv1d(Tensor x, Tensor weight, bool allow_hf32=False) -> Tensor
```

## 功能描述

因果 depthwise 一维卷积（causal depthwise Conv1D）。每个通道独立计算，不做通道间求和：

$$
y[b,h,t] = \sum_{j=0}^{K-1} x[b,h,\,t-j] \cdot k[h,j], \qquad t = 0,1,\dots,L-1
$$

其中 $x[b,h,\tau] = 0$ 当 $\tau < 0$（因果补零）。

支持反向传播。

## 参数说明

- `x(Tensor)`：输入序列，数据类型为`float32`，shape 为`(B, H, L)`，其中`B`为 batch size，`H`为通道数，`L`为序列长度。
- `weight(Tensor)`：卷积核，数据类型为`float32`，shape 为`(H, K)`，其中`H`为通道数（须与`x`一致），`K`为卷积核长度。
- `allow_hf32(bool)`：可选参数，默认为`False`。控制算子内部的矩阵乘是否使用 hf32 作为中间计算格式。仅在<term>Ascend 950PR&950DT系列产品</term>上生效，其余型号该参数被忽略。

## 返回值

- `output(Tensor)`：输出张量，数据类型为`float32`，shape 为`(B, H, L)`，与输入`x`同 shape。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>

## 约束说明

1. `x`与`weight`的数据类型均只支持`float32`。
2. `x`的通道数与`weight`的通道数必须一致，即 `x.size(1) == weight.size(0)`。
3. 卷积核长度不得超过序列长度，即 $K \le L$。
4. `allow_hf32=True`时矩阵乘使用 hf32 中间计算格式，吞吐更高但**尾数精度低于严格 float32**。默认关闭，即默认走严格 float32。该参数仅在<term>Ascend 950PR&950DT系列产品</term>上生效。

## 调用示例

```python
import torch
import torch_npu
from mx_driving import npu_fft_conv1d, FftConv1dFunction

B, H, L, K = 4, 32, 4096, 512

x = torch.randn(B, H, L).npu()
weight = torch.randn(H, K).npu()
x.requires_grad_()
weight.requires_grad_()

# 位置传参
output = npu_fft_conv1d(x, weight)

# 关键字传参
output = npu_fft_conv1d(x=x, weight=weight, allow_hf32=False)

output.backward(torch.ones_like(output))
```
