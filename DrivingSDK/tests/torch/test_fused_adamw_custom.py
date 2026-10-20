# pylint: disable=R0801
import math
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def gen_inputs(shape, dtype):
    param = torch.randn(shape, dtype=dtype)
    grad = torch.randn(shape, dtype=dtype)
    exp_avg = torch.randn(shape, dtype=dtype)
    exp_avg_sq = torch.randn(shape, dtype=dtype)
    max_exp_avg_sq = torch.randn(shape, dtype=dtype)
    return param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq


@golden_data_cache(__file__)
def gen_golden(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, weight_decay, eps, step, amsgrad):
    param = param.clone().float()
    exp_avg = exp_avg.clone().float()
    exp_avg_sq = exp_avg_sq.clone().float()
    max_exp_avg_sq = max_exp_avg_sq.clone().float()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    param.mul_(1 - lr * weight_decay)
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    if amsgrad:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    else:
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1
    param.addcdiv_(exp_avg, denom, value=-step_size)

    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


class TestAdamParamUpdate(TestCase):
    def test_adamw_param_update(self, device="npu"):
        for shape in [1, 100, 10000, 1000000, [4, 3, 256, 256]]:
            for dtype in [torch.float, torch.float16, torch.bfloat16]:
                for amsgrad in [True, False]:
                    param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq = gen_inputs(shape, dtype)
                    lr = 0.001
                    beta1 = 0.9
                    beta2 = 0.999
                    weight_decay = 0.01
                    eps = 1e-8
                    step = 1000

                    param_npu = param.npu()
                    grad_npu = grad.npu()
                    exp_avg_npu = exp_avg.npu()
                    exp_avg_sq_npu = exp_avg_sq.npu()
                    max_exp_avg_sq_npu = max_exp_avg_sq.npu()

                    param_cpu, exp_avg_cpu, exp_avg_sq_cpu, max_exp_avg_sq_cpu = gen_golden(
                        param_npu,
                        grad_npu,
                        exp_avg_npu,
                        exp_avg_sq_npu,
                        max_exp_avg_sq_npu,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        step,
                        amsgrad,
                    )

                    mx_driving.fused_adamw_custom(
                        param_npu,
                        grad_npu,
                        exp_avg_npu,
                        exp_avg_sq_npu,
                        max_exp_avg_sq_npu,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        step,
                        amsgrad,
                    )

                    self.assertRtolEqual(param_npu.float().cpu().numpy(), param_cpu.float().cpu().numpy(), prec=1e-2)
                    self.assertRtolEqual(
                        exp_avg_npu.float().cpu().numpy(), exp_avg_cpu.float().cpu().numpy(), prec=1e-2
                    )
                    self.assertRtolEqual(
                        exp_avg_sq_npu.float().cpu().numpy(), exp_avg_sq_cpu.float().cpu().numpy(), prec=1e-2
                    )
                    self.assertRtolEqual(
                        max_exp_avg_sq_npu.float().cpu().numpy(), max_exp_avg_sq_cpu.float().cpu().numpy(), prec=1e-2
                    )


if __name__ == "__main__":
    run_tests()
