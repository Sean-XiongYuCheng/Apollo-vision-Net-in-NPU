"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch

import mx_driving._C


def fused_adamw_custom(
    combined_param: torch.Tensor,
    combined_grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    max_exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
    amsgrad: bool,
):
    """
    Adam parameter update (inplace operation).

    Args:
        combined_param: Parameter tensor to be updated (inplace).
            Supported dtypes: torch.float32, torch.float16, torch.bfloat16
        combined_grad: Gradient tensor.
            Supported dtypes: torch.float32, torch.float16, torch.bfloat16
        exp_avg: First moment estimate (inplace updated).
            Supported dtypes: torch.float32, torch.float16, torch.bfloat16
        exp_avg_sq: Second moment estimate (inplace updated).
            Supported dtypes: torch.float32, torch.float16, torch.bfloat16
        max_exp_avg_sq: Maximum second moment estimate for AMSGrad (inplace updated).
            Supported dtypes: torch.float32, torch.float16, torch.bfloat16
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient
        eps: Small constant for numerical stability
        step: Current optimization step count, used for bias correction of first and second moment estimates
        amsgrad: Whether to use AMSGrad variant

    Note:
        This is an inplace operation. The input tensors (combined_param, exp_avg,
        exp_avg_sq, max_exp_avg_sq) will be modified directly.
    """
    if torch.numel(combined_param) == 0:
        return

    if isinstance(step, torch.Tensor):
        raise TypeError("step only support scalar input")

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if not amsgrad:
        max_exp_avg_sq = combined_param.new_zeros(0)

    mx_driving._C.adamw_param_update(
        combined_param,
        combined_grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        bias_correction1,
        bias_correction2,
        amsgrad,
    )
