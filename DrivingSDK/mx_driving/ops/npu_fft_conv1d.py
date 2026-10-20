"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04
Modification Description:
Modification 1. Add support for Ascend NPU
"""

from typing import Tuple

import torch
from torch.autograd import Function

import mx_driving._C


class FftConv1dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        allow_hf32: bool = False,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.allow_hf32 = allow_hf32
        return mx_driving._C.npu_fft_conv1d(x, weight, allow_hf32)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        # pylint: disable=unpacking-non-sequence, too-many-function-args
        grad_x, grad_weight = mx_driving._C.npu_fft_conv1d_grad(grad, x, weight, ctx.allow_hf32)
        return grad_x, grad_weight, None


def npu_fft_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_hf32: bool = False,
) -> torch.Tensor:
    return FftConv1dFunction.apply(x, weight, allow_hf32)
