# Copyright (c) 2024, Huawei Technologies.All rights reserved.
# Copyright 2021 Yan Yan
"""Compare results between different algos:
CPU: simple gather-mm-scatter
Native: Fused gather-mm-scatter
ImplicitGemm: implicit gemm
"""

# pylint: disable=duplicate-code
import numpy as np
import torch
from torch import nn
from torch_npu.testing.testcase import TestCase, run_tests
from data_cache import golden_data_cache
from test_subm_sparse_conv3d import get_golden_output as forward_func
from mx_driving.spconv import SparseSequential, SparseConvTensor, SubMConv3d
from cv_fused_double_benchmark_compare import CvFusedDoubleBenchmarkAccuracyCompare
import os

torch.npu.conv.allow_hf32 = False


CASES = dict(
    test_3x3x3_fp32=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_large_point1=dict(
        num_points=[200000],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_large_point2=dict(
        num_points=[200000],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_model_5x5x5_fp32=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=5,
        dtype=torch.float32,
    ),
    test_model_5x5x5_fp16=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=5,
        dtype=torch.float16,
    ),
    test_3x3x3_fp16=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_model_large_channels1=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=256,
        out_channels=512,
        kernel_size=5,
        dtype=torch.float16,
    ),
    test_model_large_channels2=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_model_large_channels3=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_model_large_channels4=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=256,
        out_channels=512,
        kernel_size=5,
        dtype=torch.float32,
    ),
    test_3x3x3_big_case_large_channels1=dict(
        num_points=[51670],
        out_spatial_shape=[128, 128, 128],
        in_channels=1024,
        out_channels=1024,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_3x3x3_big_case_large_channels1_fp16=dict(
        num_points=[51670],
        out_spatial_shape=[128, 128, 128],
        in_channels=1024,
        out_channels=1024,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_3x3x3_unaligned_channel_case1=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=5,
        out_channels=17,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_3x3x3_unaligned_channel_case1_fp16=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=5,
        out_channels=17,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_3x3x3_unaligned_channel_case2=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=5,
        out_channels=256,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_3x3x3_unaligned_channel_case2_fp16=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=5,
        out_channels=256,
        kernel_size=3,
        dtype=torch.float16,
    ),
    test_inchannel8=dict(
        num_points=[61557],
        out_spatial_shape=[1440, 1440, 41],
        in_channels=8,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float32,
    ),
    test_int32_boundary=dict(
        num_points=[61557] * 8,
        out_spatial_shape=[1440, 1440, 180],
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dtype=torch.float32,
    ),
)


@golden_data_cache(__file__)
def generate_sparse_data(num_points, spatial_shape, in_channels, out_channels):
    assert len(spatial_shape) == 3, f"Expected 3D spatial shape, got {len(spatial_shape)}"
    bs, H, W, D = len(num_points), *spatial_shape  # noqa: F841
    total_points = sum(num_points)
    total_size = H * W * D

    features = np.random.uniform(-5, 5, (total_points, in_channels))
    grad_out = np.random.uniform(-5, 5, (total_points, out_channels))

    indices = []
    for batch_idx, num_point in enumerate(num_points):
        assert num_point <= total_size, f'num_points {num_point} exceeds total spatial size {total_size}'

        batch_indices = []
        batch_indices.append(np.ones((2 * num_point, 1)) * batch_idx)
        for spatial_size in spatial_shape:
            idx = np.random.uniform(0, spatial_size, (2 * num_point, 1)).astype(np.int32)
            batch_indices.append(idx)

        batch_indices = np.concatenate(batch_indices, axis=1)
        idx_unique = np.unique(batch_indices, axis=0)
        indices.append(idx_unique[:num_point])

    indices = np.concatenate(indices, axis=0)

    return torch.from_numpy(features).float(), torch.from_numpy(indices).int(), torch.from_numpy(grad_out).float()


@golden_data_cache(__file__)
def get_golden_output(
    features,
    indices,
    weight,
    bias,
    batch_size,
    in_channels,
    out_channels,
    kernel_size,
    out_spatial_shape,
    grad_out,
    dtype,
):
    _, indices_offset, img2col_mat = forward_func(
        features, indices, weight, bias, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape
    )
    features_grad, weight_grad = subm_sparse_conv3d_grad_cpu(
        grad_out, img2col_mat, indices_offset, weight, kernel_size, in_channels, dtype
    )
    return features_grad, weight_grad


@golden_data_cache(__file__)
def subm_sparse_conv3d_grad_cpu(grad_out, img2col_mat, indices_offset, weight, kernel_size, in_channels, dtype):
    dtype = torch.float32
    grad_out, img2col_mat, indices_offset, weight = (
        grad_out.to(dtype).cpu(),
        img2col_mat.to(dtype).cpu(),
        indices_offset.cpu(),
        weight.to(dtype).cpu(),
    )

    N, out_channels = grad_out.shape
    weight_grad = img2col_mat.reshape(N, -1).transpose(1, 0) @ grad_out
    img2col_mat_grad = grad_out @ weight.reshape(-1, out_channels).T
    indices_offset = indices_offset.reshape(-1)
    img2col_mat_grad = img2col_mat_grad.reshape(-1, in_channels)

    weight_shape = weight.shape
    weight_grad = weight_grad.view(weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4])

    mask = indices_offset != -1
    valid_indices = torch.nonzero(mask).view(-1)
    indices_offset = torch.index_select(indices_offset, 0, valid_indices)
    img2col_mat_grad = torch.index_select(img2col_mat_grad, 0, valid_indices)

    features_grad = torch.zeros(N, in_channels, dtype=grad_out.dtype, device=grad_out.device)

    features_grad.scatter_add_(0, indices_offset[:, None].broadcast_to((-1, in_channels)).long(), img2col_mat_grad)
    return features_grad, weight_grad


def single_benchmark_get_output(num_points, batch_size, in_channels, out_channels, kernel_size, spatial_shape, dtype):
    features, indices, grad_out = generate_sparse_data(num_points, spatial_shape, in_channels, out_channels)
    features, indices, grad_out = features.to(dtype).npu(), indices.npu(), grad_out.to(dtype).npu()
    net = SubMConv3d(in_channels, out_channels, kernel_size).npu()
    net.weight.data = net.weight.data.to(dtype)
    net.bias.data = net.bias.data.to(dtype) * 0.01
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    features.requires_grad = True
    net.weight.requires_grad = True

    features_grad, weight_grad = get_golden_output(
        features,
        indices,
        net.weight.data,
        net.bias.data,
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        spatial_shape,
        grad_out,
        dtype,
    )
    res = net(x).features
    res.backward(grad_out)
    return features.grad.cpu(), net.weight.grad.cpu(), features_grad.cpu(), weight_grad.cpu()


def double_benchmark_get_output(
    num_points, batch_size, in_channels, out_channels, kernel_size, spatial_shape, dtype, gpu_data
):
    features = gpu_data['features'].detach().clone()
    indices = gpu_data['indices'].detach().clone()
    grad_out = gpu_data['grad_out'].detach().clone()
    gpu_net = gpu_data['net']

    features, indices, grad_out = features.to(dtype).npu(), indices.npu(), grad_out.to(dtype).npu()
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    net = SubMConv3d(in_channels, out_channels, kernel_size).npu()

    net.weight.data = gpu_net.weight.data
    net.bias.data = gpu_net.bias.data
    features.requires_grad = True
    net.weight.requires_grad = True

    res = net(x).features
    res.backward(grad_out)
    return features.grad.cpu(), net.weight.grad.cpu()


def get_output(num_points, batch_size, in_channels, out_channels, kernel_size, spatial_shape, dtype):
    features, indices, grad_out = generate_sparse_data(num_points, spatial_shape, in_channels, out_channels)
    features, indices, grad_out = features.to(dtype).npu(), indices.npu(), grad_out.to(dtype).npu()
    net = SubMConv3d(in_channels, out_channels, kernel_size).npu()
    net.weight.data = net.weight.data.to(dtype)
    net.bias.data = net.bias.data.to(dtype) * 0.01
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    features.requires_grad = True
    net.weight.requires_grad = True

    res = net(x).features
    res.backward(grad_out)

    return features.grad.cpu(), net.weight.grad.cpu()


class TestSubmSparseConv3dGrad(TestCase):
    def test_model_case(self):
        gpu_out_path = os.getenv("GPU_OUT_PATH", None)
        double_benchmark_flag = (gpu_out_path is not None) and os.path.exists(gpu_out_path)
        name = 'test_subm_sparse_conv3d_grad'

        for case_name, case_value in CASES.items():
            num_points, out_spatial_shape, in_channels, out_channels, kernel_size, dtype = case_value.values()
            batch_size = len(num_points)

            if double_benchmark_flag:
                filepath = os.path.join(gpu_out_path, name, f'{case_name}.pth')
                gpu_data = torch.load(filepath, map_location='npu')
                gpu_features_grad = gpu_data['features_grad'].detach().clone().cpu()
                gpu_weight_grad = gpu_data['weight_grad'].detach().clone().cpu()
                cpu_features_grad = gpu_data['cpu_features_grad'].detach().clone().cpu()  # 升精度
                cpu_weight_grad = gpu_data['cpu_weight_grad'].detach().clone().cpu()  # 升精度

                npu_features_grad, npu_weight_grad = double_benchmark_get_output(
                    num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape, dtype, gpu_data
                )
                compare = CvFusedDoubleBenchmarkAccuracyCompare(
                    [npu_features_grad, npu_weight_grad],
                    [gpu_features_grad, gpu_weight_grad],
                    [cpu_features_grad, cpu_weight_grad],
                )
                res = compare.run()
                assert "False" not in res, f"Accuracy check failed for model case {case_name}"

            else:
                npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = single_benchmark_get_output(
                    num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape, dtype
                )

                self.assertRtolEqual(npu_features_grad, cpu_features_grad.to(dtype), 1e-2, 1e-2)
                self.assertRtolEqual(npu_weight_grad, cpu_weight_grad.to(dtype), 1e-2, 1e-2)

    def test_missing_branch(self):
        from mx_driving.modules.sparse_modules import ToDense, RemoveGrid, _mean_update
        from mx_driving.modules.sparse_conv import get_deconv_output_size

        num_points = [100]
        spatial_shape = [30, 30, 30]
        in_channels = 32
        out_channels = 32
        kernel_size = 3
        dtype = torch.float32
        batch_size = len(num_points)

        features, indices, grad_out = generate_sparse_data(num_points, spatial_shape, in_channels, out_channels)
        features, indices, grad_out = features.to(dtype).npu(), indices.npu(), grad_out.to(dtype).npu()
        net = SubMConv3d(in_channels, out_channels, kernel_size).npu()
        net.weight.data = net.weight.data.to(dtype)
        net.bias.data = net.bias.data.to(dtype) * 0.01
        features.requires_grad = True
        net.weight.requires_grad = True

        x = SparseConvTensor(features, indices, spatial_shape, batch_size)
        SeqNet = SparseSequential(net)
        x.dense(channels_first=False)
        SeqNet.add(nn.Sequential())
        SeqNet = SeqNet.fused()
        SeqRes = SeqNet(x).features
        SeqRes.backward(grad_out)
        cpu_features_grad, cpu_weight_grad = get_golden_output(
            features,
            indices,
            net.weight.data,
            net.bias.data,
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            spatial_shape,
            grad_out,
            dtype,
        )
        npu_features_grad, npu_weight_grad = features.grad.cpu(), net.weight.grad.cpu()

        testnet = SubMConv3d(in_channels, out_channels, kernel_size, bias=False)
        get_deconv_output_size(spatial_shape, [kernel_size] * 3, [1, 1, 1], [0, 0, 0], 0, [0, 0, 0])
        _mean_update([10, 20], [4, 8], 3)
        testnet.mode = None
        testnet.reset_parameters()
        DenseRes = ToDense()(x)  # noqa: F841
        GridRes = RemoveGrid()(x)  # noqa: F841

        self.assertRtolEqual(npu_features_grad, cpu_features_grad.to(dtype), 1e-2, 1e-2)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad.to(dtype), 1e-2, 1e-2)

    def test_huge_case(self):
        # only for test
        num_points = [504420]
        spatial_shape = [320, 640, 640]
        in_channels = 256
        out_channels = 256
        kernel_size = 3

        get_output(num_points, len(num_points), in_channels, out_channels, kernel_size, spatial_shape, torch.float32)
        get_output(num_points, len(num_points), in_channels, out_channels, kernel_size, spatial_shape, torch.float16)


if __name__ == "__main__":
    np.random.seed(100)
    run_tests()
