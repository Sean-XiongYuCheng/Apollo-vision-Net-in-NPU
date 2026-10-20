"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-07-03
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import functools
from typing import Any, Callable

import torch
from torch.autograd import Function
from torch import Tensor
import mx_driving._C
import mx_driving


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    cache = []
    cache_size = 4

    def _equal(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a is b
        return a == b

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        for cached_args, cached_kwargs, cached_result in cache:
            if all(_equal(a, b) for a, b in zip(args, cached_args)) and all(
                k in cached_kwargs and _equal(v, cached_kwargs[k]) for k, v in kwargs.items()
            ):
                return cached_result
        result = fn(*args, **kwargs)
        cache.insert(0, (args, kwargs, result))
        if len(cache) > cache_size:
            cache = cache[:cache_size]
        return result

    return wrapper


@tensor_cache
def _is_scattered(index: torch.Tensor) -> bool:
    return torch.unique(index).numel() == index.numel()


class GraphSoftmax(Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor) -> Tensor:
        N = int(index.max()) + 1
        idx = index.to(torch.int32)
        softmaxResult = mx_driving._C.graph_softmax(src, idx, N)
        ctx.save_for_backward(idx, softmaxResult)
        ctx.is_scattered = _is_scattered(idx)
        return softmaxResult

    @staticmethod
    def backward(ctx, grad_output):
        index, softmax_out = ctx.saved_tensors
        N = int(index.max()) + 1
        grad_src = mx_driving._C.graph_softmax_grad(index, softmax_out, grad_output, N, ctx.is_scattered)  # pylint: disable=too-many-function-args
        return grad_src, None


graph_softmax = GraphSoftmax.apply
