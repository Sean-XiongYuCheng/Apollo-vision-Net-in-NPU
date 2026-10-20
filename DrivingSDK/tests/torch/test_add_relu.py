import numpy as np
import torch
import torch.nn.functional as F
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.fused


@golden_data_cache(__file__)
def gen_inputs(shape, dtype):
    x = np.random.uniform(-1, 1, shape).astype(dtype)
    x = torch.from_numpy(x)
    y = np.random.uniform(-1, 1, shape).astype(dtype)
    y = torch.from_numpy(y)
    grad = np.random.uniform(-1, 1, shape).astype(dtype)
    grad = torch.from_numpy(grad)
    return x, y, grad


@golden_data_cache(__file__)
def gen_cpu_outputs(x, y, grad):
    x_float = x.float().detach()
    y_float = y.float().detach()
    x_float.requires_grad_()
    y_float.requires_grad_()

    output = F.relu(x_float + y_float)
    output.backward(grad.float())

    return output, x_float.grad, y_float.grad


class TestAddRelu(TestCase):
    def test_npu_add_relu_three_dim(self, device="npu"):
        x, y, grad = gen_inputs([1, 100, 3], np.float32)
        cpu_result, grad_x_cpu, grad_y_cpu = gen_cpu_outputs(x, y, grad)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu())
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())

        x_npu, y_npu, grad_npu = x.npu(), y.npu(), grad.npu()
        x_npu.requires_grad_()
        y_npu.requires_grad_()
        result = mx_driving.npu_add_relu(x_npu, y_npu)
        result.backward(grad_npu)
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())
        self.assertRtolEqual(x_npu.grad.cpu().half().numpy(), grad_x_cpu.half().numpy())
        self.assertRtolEqual(y_npu.grad.cpu().half().numpy(), grad_y_cpu.half().numpy())

    def test_npu_add_relu_large_number(self, device="npu"):
        x, y, grad = gen_inputs([18, 256, 232, 100], np.float32)
        cpu_result, grad_x_cpu, grad_y_cpu = gen_cpu_outputs(x, y, grad)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu())
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())

        x_npu, y_npu, grad_npu = x.npu(), y.npu(), grad.npu()
        x_npu.requires_grad_()
        y_npu.requires_grad_()
        result = mx_driving.npu_add_relu(x_npu, y_npu)
        result.backward(grad_npu)
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())
        self.assertRtolEqual(x_npu.grad.cpu().half().numpy(), grad_x_cpu.half().numpy())
        self.assertRtolEqual(y_npu.grad.cpu().half().numpy(), grad_y_cpu.half().numpy())

    def test_npu_add_relu_fp16_large_number(self, device="npu"):
        x, y, grad = gen_inputs([18, 256, 232, 100], np.float16)
        cpu_result, grad_x_cpu, grad_y_cpu = gen_cpu_outputs(x, y, grad)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu())
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())

        x_npu, y_npu, grad_npu = x.npu(), y.npu(), grad.npu()
        x_npu.requires_grad_()
        y_npu.requires_grad_()
        result = mx_driving.npu_add_relu(x_npu, y_npu)
        result.backward(grad_npu)
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())
        self.assertRtolEqual(x_npu.grad.cpu().half().numpy(), grad_x_cpu.half().numpy())
        self.assertRtolEqual(y_npu.grad.cpu().half().numpy(), grad_y_cpu.half().numpy())

    def test_npu_add_relu_fp16_small_case(self, device="npu"):
        x, y, grad = gen_inputs([18], np.float16)
        cpu_result, grad_x_cpu, grad_y_cpu = gen_cpu_outputs(x, y, grad)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu())
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())

        x_npu, y_npu, grad_npu = x.npu(), y.npu(), grad.npu()
        x_npu.requires_grad_()
        y_npu.requires_grad_()
        result = mx_driving.npu_add_relu(x_npu, y_npu)
        result.backward(grad_npu)
        self.assertRtolEqual(result.detach().cpu().half().numpy(), cpu_result.detach().half().numpy())
        self.assertRtolEqual(x_npu.grad.cpu().half().numpy(), grad_x_cpu.half().numpy())
        self.assertRtolEqual(y_npu.grad.cpu().half().numpy(), grad_y_cpu.half().numpy())


if __name__ == "__main__":
    run_tests()
