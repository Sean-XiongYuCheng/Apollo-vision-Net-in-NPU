# sigmoid_focal_loss

## Interface Prototype

```python
mx_driving.sigmoid_focal_loss(Tensor logit, Tensor target, float gamma=2, float alpha=0.25, Tensor weight=None, str reduction='mean') -> Tensor
```

## Description

First computes the sigmoid value of each element in the input `logit`, and then computes the focal loss between the sigmoid values and the class target values. This function is consistent with `mmcv`'s `sigmoid_focal_loss`.

## Parameters

- `logit` (Tensor): Classification predictions for all samples. The data type is `float32`. The shape is `[N, C]`, where `N` is the number of samples and `C` is the number of classes.
- `target` (Tensor): Classification target values for all samples. The data type is `int64`. The shape is `[N]`.
- `gamma` (float): Hyperparameter used to balance easy and hard samples. The default value is 2.0.
- `alpha` (float): Hyperparameter used to balance positive and negative samples. The default value is 0.25.
- `weight` (Tensor): Weight coefficients for all classes. The data type is `float32`. The shape is `[C]`.
- `reduction` (str): Reduction method. The data type is `str`. The default value is `mean`.

## Returns

- `output` (Tensor): Focal loss for each element in the input. The data type is `float32`. The shape is `[N, C]`.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950 products</term>

## Example

```python
import torch, torch_npu
from mx_driving import sigmoid_focal_loss
logit = torch.rand(1800, 10, dtype=torch.float32, device='npu') * 10 - 5
target = torch.randint(low=0, high=10, size=(1800,), dtype=torch.int64, device='npu')
weight = torch.rand(1800, dtype=torch.float32, device='npu') * 10 - 5
logit.requires_grad = True
output = sigmoid_focal_loss(logit, target, 2.0, 0.25, weight, 'mean')
output.backward()
```
