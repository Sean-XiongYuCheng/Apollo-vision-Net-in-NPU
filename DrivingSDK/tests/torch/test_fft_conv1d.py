import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


@golden_data_cache(__file__)
def gen_inputs(shape, weight_shape, dtype):
    x = np.random.uniform(-1, 1, shape).astype(dtype)
    x = torch.from_numpy(x)
    weight = np.random.uniform(-1, 1, weight_shape).astype(dtype)
    weight = torch.from_numpy(weight)
    return x, weight


@golden_data_cache(__file__)
def gen_cpu_outputs_fft(x, weight):
    """
    x:
        [B, C, L]

    weight:
        [C, 1, K]，对应 groups=C 的 depthwise conv1d
        或 [C, K]

    output:
        [B, C, L]
    """
    x_float = x.float().detach().requires_grad_(True)
    weight_float = weight.float().detach().requires_grad_(True)

    batch, channels, input_length = x_float.shape

    # [C, 1, K] -> [C, K]
    if weight_float.dim() == 3:
        if weight_float.shape[0] != channels:
            raise ValueError(f"weight.shape[0] 应等于通道数 {channels}，实际为 {weight_float.shape[0]}")
        if weight_float.shape[1] != 1:
            raise ValueError("groups=channels 的逐通道卷积要求 weight shape 为 [C, 1, K]")
        kernel = weight_float[:, 0, :]
    elif weight_float.dim() == 2:
        if weight_float.shape[0] != channels:
            raise ValueError(f"weight.shape[0] 应等于通道数 {channels}，实际为 {weight_float.shape[0]}")
        kernel = weight_float
    else:
        raise ValueError("weight 必须是 [C, 1, K] 或 [C, K]")

    kernel_size = kernel.shape[-1]

    # 线性卷积至少需要 L + K - 1 个点，防止循环卷积混叠
    fft_size = input_length + kernel_size - 1

    # 可以选择补到 2 的幂，提高 FFT 执行效率
    fft_size = 1 << (fft_size - 1).bit_length()

    # x_f: [B, C, fft_size // 2 + 1]
    x_f = torch.fft.rfft(
        x_float,
        n=fft_size,
        dim=-1,
    )

    # kernel_f: [C, fft_size // 2 + 1]
    kernel_f = torch.fft.rfft(
        kernel,
        n=fft_size,
        dim=-1,
    )

    # [B, C, F] * [1, C, F]
    output_f = x_f * kernel_f.unsqueeze(0)

    # 完整线性卷积结果：[B, C, fft_size]
    output_full = torch.fft.irfft(
        output_f,
        n=fft_size,
        dim=-1,
    )

    # 因果卷积只取前 L 个位置
    output = output_full[..., :input_length]

    return output


@golden_data_cache(__file__)
def gen_grad(shape, dtype):
    grad = np.random.uniform(-1, 1, shape).astype(dtype)
    return torch.from_numpy(grad)


@golden_data_cache(__file__)
def gen_cpu_grads_fft(x, weight, grad):
    """rfft 路径的参考反向，与 gen_cpu_outputs_fft 对应（K 很大时用）。

    K=8192 这类 shape 下直接卷积是 O(L*K)，CPU 上跑不动；torch.fft 在 PyTorch
    里是可微的，autograd 直接穿过频域，得到的就是同一组梯度。
    """
    x_float = x.float().detach().requires_grad_(True)
    weight_float = weight.float().detach().requires_grad_(True)

    _, channels, input_length = x_float.shape
    kernel = weight_float[:, 0, :] if weight_float.dim() == 3 else weight_float
    kernel_size = kernel.shape[-1]

    fft_size = input_length + kernel_size - 1
    fft_size = 1 << (fft_size - 1).bit_length()

    x_f = torch.fft.rfft(x_float, n=fft_size, dim=-1)
    kernel_f = torch.fft.rfft(kernel, n=fft_size, dim=-1)
    output_full = torch.fft.irfft(x_f * kernel_f.unsqueeze(0), n=fft_size, dim=-1)
    output = output_full[..., :input_length]

    output.backward(grad.float())
    grad_weight = weight_float.grad
    if grad_weight.dim() == 3:
        grad_weight = grad_weight.squeeze(1)
    return x_float.grad, grad_weight


class TestFftConv1d(TestCase):
    def test_npu_fft_conv1d_large(self, device="npu"):
        x, weight = gen_inputs([32, 32, 8193], [32, 1, 8192], np.float32)
        cpu_result = gen_cpu_outputs_fft(x, weight)
        x_npu, weight_npu = x.npu(), weight.squeeze(1).npu()
        x_npu.requires_grad_()
        result = mx_driving.npu_fft_conv1d(x_npu, weight_npu)
        self.assertRtolEqual(result.cpu(), cpu_result)

    def test_npu_fft_conv1d_small(self, device="npu"):
        x, weight = gen_inputs([1, 32, 1023], [32, 1, 5], np.float32)
        cpu_result = gen_cpu_outputs_fft(x, weight)
        x_npu, weight_npu = x.npu(), weight.squeeze(1).npu()
        x_npu.requires_grad_()
        result = mx_driving.npu_fft_conv1d(x_npu, weight_npu)
        self.assertRtolEqual(result.cpu(), cpu_result)


class TestFftConv1dBackward(TestCase):
    """反向语义（推导见 kernels/fft_conv1d/op_kernel/fft_conv1d_grad.cpp）：

        grad_x[b,h,tau] = sum_j grad[b,h,tau+j] * weight[h,j]      grad[t]=0 当 t>=L
        grad_w[h,j]     = sum_b sum_tau x[b,h,tau] * grad[b,h,tau+j]

    grad_w 要对 batch 求和，是反向唯一打破"行间独立"的地方；设备侧因此按通道
    切分、逐 batch 在频域累加，每个通道只做一次逆变换。
    """

    def test_npu_fft_conv1d_backward_float32(self, device="npu"):
        shape, weight_shape = [2, 1024, 8193], [1024, 1, 8192]
        x, weight = gen_inputs(shape, weight_shape, np.float32)
        grad = gen_grad(shape, np.float32)
        grad_x_cpu, grad_w_cpu = gen_cpu_grads_fft(x, weight, grad)
        x_npu = x.npu().detach().requires_grad_()
        weight_npu = weight.squeeze(1).npu().detach().requires_grad_()
        result = mx_driving.npu_fft_conv1d(x_npu, weight_npu)
        result.backward(grad.npu())
        self.assertRtolEqual(x_npu.grad.cpu(), grad_x_cpu)
        self.assertRtolEqual(weight_npu.grad.cpu(), grad_w_cpu, 1e-3, 1e-3)


if __name__ == "__main__":
    run_tests()
