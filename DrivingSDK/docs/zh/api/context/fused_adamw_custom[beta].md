# fused_adamw_custom[beta]

## 接口原型

```python
mx_driving.fused_adamw_custom(Tensor combined_param, Tensor combined_grad, Tensor exp_avg, Tensor exp_avg_sq, Tensor max_exp_avg_sq, float lr, float beta1, float beta2, float weight_decay, float eps, int step, bool amsgrad) -> None
```

## 功能描述

AdamW参数更新，原地操作。该算子实现了AdamW优化器的参数更新逻辑，用于模型训练过程中的参数优化。

计算公式如下：

$$
g_t = \nabla f(\theta_{t-1}) + \lambda \theta_{t-1}
$$

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{v}_t = \begin{cases} \max(\hat{v}_{t-1}, v_t) & \text{if amsgrad=True} \\ \hat{v}_t, & \text{if amsgrad=False} \end{cases}
$$

$$
denom = \begin{cases} \sqrt{\hat{v}_t / (1 - \beta_2^t)} + \epsilon & \text{if amsgrad=True} \\ \sqrt{v_t / (1 - \beta_2^t)} + \epsilon & \text{if amsgrad=False} \end{cases}
$$

$$
\hat{m}_t = m_t / (1 - \beta_1^t)
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{denom}
$$

参数与公式符号对应关系：

- `combined_param`：$\theta$，待更新的参数
- `combined_grad`：$\nabla f(\theta_{t-1})$，原始梯度
- `exp_avg`：$m_t$，一阶矩估计
- `exp_avg_sq`：$v_t$，二阶矩估计
- `max_exp_avg_sq`：$\hat{v}_t$，最大二阶矩估计（AMSGrad变体）
- `lr`：$\eta$，学习率
- `beta1`：$\beta_1$，一阶矩的指数衰减率
- `beta2`：$\beta_2$，二阶矩的指数衰减率
- `weight_decay`：$\lambda$，权重衰减系数
- `eps`：$\epsilon$，数值稳定性常数
- `step`：$t$，当前优化步数

## 参数说明

- `combined_param(Tensor)`：待更新的参数张量，原地更新。数据类型为`float32`、`float16`或`bfloat16`。
- `combined_grad(Tensor)`：梯度张量。数据类型为`float32`、`float16`或`bfloat16`。
- `exp_avg(Tensor)`：一阶矩估计，原地更新。数据类型为`float32`、`float16`或`bfloat16`。
- `exp_avg_sq(Tensor)`：二阶矩估计，原地更新。数据类型为`float32`、`float16`或`bfloat16`。
- `max_exp_avg_sq(Tensor)`：最大二阶矩估计，用于AMSGrad变体，原地更新。数据类型为`float32`、`float16`或`bfloat16`。
- `lr(float)`：学习率。
- `beta1(float)`：一阶矩的指数衰减率。
- `beta2(float)`：二阶矩的指数衰减率。
- `weight_decay(float)`：权重衰减系数。
- `eps(float)`：数值稳定性的小常数。
- `step(int)`：当前优化步数。
- `amsgrad(bool)`：是否使用AMSGrad变体。

## 返回值

无返回值，原地操作。

## 约束说明

1. 输入张量`combined_param`、`combined_grad`、`exp_avg`、`exp_avg_sq`、`max_exp_avg_sq`的shape和数据类型需要保持一致。
2. 该算子为原地操作，输入张量`combined_param`、`exp_avg`、`exp_avg_sq`、`max_exp_avg_sq`会被直接修改。
3. 当`amsgrad`为`False`时，`max_exp_avg_sq`参数不会被使用。

## 支持的型号

- Atlas A2训练系列产品
- Atlas A3训练系列产品
- Ascend 950PR&950DT系列产品

## 调用示例

```python
import torch
import torch_npu
from mx_driving import fused_adamw_custom

# 初始化参数和梯度
param = torch.randn(1024, 1024, dtype=torch.float32).npu()
grad = torch.randn(1024, 1024, dtype=torch.float32).npu()
exp_avg = torch.zeros_like(param)
exp_avg_sq = torch.zeros_like(param)
max_exp_avg_sq = torch.zeros_like(param)

# 设置优化器参数
lr = 0.001
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.01
eps = 1e-8
step = 1
amsgrad = True

# 执行参数更新
fused_adamw_custom(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, weight_decay, eps, step, amsgrad)
```
