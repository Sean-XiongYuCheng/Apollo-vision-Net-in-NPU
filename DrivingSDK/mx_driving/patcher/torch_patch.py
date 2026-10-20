# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# pylint: disable=cyclic-import
"""Torch tensor operation patches for NPU optimization."""

from torch_npu.utils._error_code import ErrCode, pta_error
from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch


class TensorIndex(Patch):
    """Tensor indexing optimization patch for torch."""

    name = "tensor_index"
    legacy_name = "index"
    target_module = "torch"

    # pylint: disable=bad-staticmethod-argument
    @staticmethod
    def _runtime_check(self, indices, *_, **__) -> bool:
        import torch

        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.bool or indices.dim() != 1:
            return False
        if self.dim() == 1:
            return True
        if self.dim() == 2 and self.shape[0] == indices.shape[0]:
            return True
        return False

    # pylint: disable=bad-staticmethod-argument
    @staticmethod
    def _replacement(self, indices):
        import torch

        if self.dim() == 1:
            return torch.masked_select(self, indices)
        if self.dim() == 2 and self.shape[0] == indices.shape[0]:
            indices = indices.unsqueeze(1).expand(self.shape)
            return torch.masked_select(self, indices).view(-1, self.shape[1])
        return torch.masked_select(self, indices)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "torch.Tensor.__getitem__",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
        ]


class BatchMatmul(Patch):
    """Batch matmul optimization patch for torch."""

    name = "batch_matmul"
    legacy_name = "batch_matmul"
    target_module = "torch"

    @staticmethod
    def _check_shape_bmm(a, b) -> bool:
        if not hasattr(b, 'dim'):
            return False
        if not (a.dim() == b.dim() and 4 <= a.dim() <= 7):
            return False
        if not all(ad == bd or ad == 1 or bd == 1 for ad, bd in zip(a.shape[:-2], b.shape[:-2])):
            return False
        return a.shape[-2] == a.shape[-1] and a.shape[-2] == b.shape[-2] and b.shape[-1] == 1

    @staticmethod
    def _runtime_check(a, b, *_, **__) -> bool:
        return BatchMatmul._check_shape_bmm(a, b)

    @staticmethod
    def _replacement(a, b):
        from mx_driving import npu_batch_matmul

        return npu_batch_matmul(a, b)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "torch.matmul",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
            AtomicPatch(
                "torch.Tensor.matmul",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
            AtomicPatch(
                "torch.Tensor.__matmul__",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
        ]


class NpuFusedAdamwStep(Patch):
    """NpuFusedAdamw _group_step optimization patch for torch_npu."""

    name = "npu_fused_adamw_step"
    legacy_name = "npu_fused_adamw_step"
    target_module = "torch_npu"

    # pylint: disable=bad-staticmethod-argument
    @staticmethod
    def _group_step(self, group_index):
        group = self.param_groups[group_index]
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    'NpuFusedAdamW does not support sparse gradients, '
                    'please consider SparseAdam instead' + pta_error(ErrCode.NOT_SUPPORT)
                )
            state_p = self.state[p]
            state_p['step'] += 1

        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(
            combined_group_params, combined_group_grads, combined_group_param_states
        ):
            if combined_param is None or combined_grad is None:
                continue

            combined_param_state['step'] += 1

            from mx_driving import fused_adamw_custom

            fused_adamw_custom(
                combined_param,
                combined_grad,
                combined_param_state['exp_avg'],
                combined_param_state['exp_avg_sq'],
                combined_param_state['max_exp_avg_sq'],
                group['lr'],
                beta1,
                beta2,
                group['weight_decay'],
                group['eps'],
                combined_param_state['step'],
                amsgrad,
            )

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "torch_npu.optim.NpuFusedAdamW._group_step",
                cls._group_step,
            ),
        ]
