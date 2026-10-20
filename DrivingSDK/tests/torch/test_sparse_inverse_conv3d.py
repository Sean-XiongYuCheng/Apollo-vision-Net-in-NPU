# Copyright (c) 2025, Huawei Technologies.All rights reserved.

import numpy as np
import torch
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import SparseConvTensor, SparseInverseConv3d
from mx_driving.modules.sparse_structure import IndiceData


def cal_sparse_conv3d_indice_data(indices, spatial_shape, kernel_size, stride, padding):
    """
    Compute the indice cache produced by a forward sparse conv, without calling SparseConv3d.

    indices: [N, 4] CPU int32 in [batch, d, h, w] order
    Returns:
      output_indices:               int32, unique sparse-conv output coordinates
      unique_indices_offset:        int32, CSR offsets for output points
      sorted_idx_to_former_indices: int32, task IDs in [0, N * kernel_product)
      outidx_pair:                  int32, output coordinate for each task ID
      out_spatial_shape:            List[int], sparse-conv output spatial shape
    """
    indices = indices.cpu()
    k0, k1, k2 = kernel_size
    out_spatial_shape = [(spatial_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3)]

    k0_off = torch.arange(k0)
    k1_off = torch.arange(k1)
    k2_off = torch.arange(k2)
    k_offset = torch.cartesian_prod(k0_off - k0 // 2, k1_off - k1 // 2, k2_off - k2 // 2)
    k_offset = torch.cat([torch.zeros(k_offset.shape[0], 1), k_offset], dim=1)

    idx_off = (k_offset + indices[:, None, :]).double()
    idx_off[..., 1] = (idx_off[..., 1] + padding[0] - k0 // 2) / stride[0]
    idx_off[..., 2] = (idx_off[..., 2] + padding[1] - k1 // 2) / stride[1]
    idx_off[..., 3] = (idx_off[..., 3] + padding[2] - k2 // 2) / stride[2]

    valid = (
        (idx_off[..., 1] >= 0)
        & (idx_off[..., 1] < out_spatial_shape[0])
        & (idx_off[..., 2] >= 0)
        & (idx_off[..., 2] < out_spatial_shape[1])
        & (idx_off[..., 3] >= 0)
        & (idx_off[..., 3] < out_spatial_shape[2])
        & (idx_off.frac() == 0).all(dim=-1)
    )

    flat = (
        idx_off[..., 0] * (out_spatial_shape[0] * out_spatial_shape[1] * out_spatial_shape[2])
        + idx_off[..., 1] * (out_spatial_shape[1] * out_spatial_shape[2])
        + idx_off[..., 2] * out_spatial_shape[2]
        + idx_off[..., 3]
    )
    flat[~valid] = -1

    outidx_pair = idx_off.to(torch.int32)
    outidx_pair[~valid] = -1
    outidx_pair = torch.flip(outidx_pair, (-2,)).reshape(-1, 4)
    output_offset = torch.flip(flat, (-1,)).to(torch.int64).flatten()

    sentinel = torch.tensor(-1)
    sorted_idx, sorted_to_former = torch.sort(output_offset)
    diff = torch.cat([sentinel.view(1), sorted_idx]) - torch.cat([sorted_idx, sentinel.view(1)])
    unique_indices_offset = torch.nonzero(diff != 0).flatten()
    if len(unique_indices_offset) == 0:
        raise ValueError("No valid pairs found -- all input points fall outside the output spatial shape.")

    output_indices = outidx_pair[sorted_to_former[unique_indices_offset[:-1]].long()]
    return (
        output_indices.int(),
        unique_indices_offset.int(),
        sorted_to_former.int(),
        outidx_pair.int(),
        out_spatial_shape,
    )


def cpu_img2col(features_cpu, unique_indices_offset_cpu, sorted_idx_cpu, n_origin, in_channel, kernel_size_product):
    """
    CPU reference for the logical img2col scatter step used by SparseInverseConv3d.

    Mirrors the two-phase NPU kernel:
      Phase 1 (flatten CSR): input_idx[j] = i  for j in [offset[i], offset[i+1])
      Phase 2 (scatter):     output[sorted_idx[k]] = features[input_idx[k]]

    output shape: [n_origin, kernel_size_product * in_channel]
    """
    N_in = features_cpu.shape[0]
    start_offset = int(unique_indices_offset_cpu[0])
    end_offset = int(unique_indices_offset_cpu[-1])

    # Phase 1: flatten CSR -> input_idx via repeat_interleave
    counts = (unique_indices_offset_cpu[1:] - unique_indices_offset_cpu[:-1]).long()
    input_idx = torch.repeat_interleave(torch.arange(N_in), counts)

    # Phase 2: scatter via index assignment
    out_rows = sorted_idx_cpu[start_offset:end_offset].long()
    output = torch.zeros(n_origin * kernel_size_product, in_channel, dtype=torch.float32)
    output[out_rows] = features_cpu[input_idx].float()
    return output.view(n_origin, kernel_size_product * in_channel)


def cpu_sparse_inverse_conv3d_backward(
    grad_out_cpu, features_cpu, weight_cpu, unique_indices_offset_cpu, sorted_idx_cpu, n_origin, kernel_size_product
):
    """CPU reference for SparseInverseConv3d backward."""
    in_channel = features_cpu.shape[1]
    output_img2col = cpu_img2col(
        features_cpu,
        unique_indices_offset_cpu,
        sorted_idx_cpu,
        n_origin,
        in_channel,
        kernel_size_product,
    )
    weight_flat = weight_cpu.reshape(-1, weight_cpu.shape[-1])
    grad_img2col = grad_out_cpu @ weight_flat.transpose(0, 1)

    counts = (unique_indices_offset_cpu[1:] - unique_indices_offset_cpu[:-1]).long()
    input_idx = torch.repeat_interleave(torch.arange(features_cpu.shape[0]), counts)
    start_offset = int(unique_indices_offset_cpu[0])
    end_offset = int(unique_indices_offset_cpu[-1])
    out_rows = sorted_idx_cpu[start_offset:end_offset].long()

    grad_features = torch.zeros_like(features_cpu)
    grad_features.index_add_(0, input_idx, grad_img2col.reshape(-1, in_channel)[out_rows])
    grad_weight = output_img2col.transpose(0, 1) @ grad_out_cpu
    grad_weight = grad_weight.reshape(weight_cpu.shape)
    return grad_features, grad_weight


def make_random_indices(spatial_shape, num_points_per_batch, batch_size):
    """Generate random sparse indices in [batch, d, h, w] format (CPU int32)."""
    all_coords = np.stack(np.meshgrid(*[np.arange(s) for s in spatial_shape]), axis=-1)
    all_coords = all_coords.reshape(-1, len(spatial_shape))
    rows = []
    for b in range(batch_size):
        perm = np.random.permutation(len(all_coords))[:num_points_per_batch]
        pts = all_coords[perm]
        batch_col = np.full((len(pts), 1), b, dtype=np.int32)
        rows.append(np.hstack([batch_col, pts]))  # [batch, d, h, w]
    return torch.from_numpy(np.vstack(rows).astype(np.int32))


class TestSparseInverseConv3dCpuRef(TestCase):
    def _run_case(
        self,
        spatial_shape,
        feature_num,
        batch_size,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        seed=0,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        ks = [kernel_size] * 3 if isinstance(kernel_size, int) else list(kernel_size)
        st = [stride] * 3 if isinstance(stride, int) else list(stride)
        pd = [padding] * 3 if isinstance(padding, int) else list(padding)
        kernel_product = ks[0] * ks[1] * ks[2]

        # 1. Origin indices (forward conv input = inverse conv output)
        origin_indices = make_random_indices(spatial_shape, feature_num, batch_size)
        N_in = origin_indices.shape[0]

        # 2. Compute indice cache without running SparseConv3d
        (
            output_indices,
            unique_indices_offset,
            sorted_idx,
            outidx_pair,
            out_spatial_shape,
        ) = cal_sparse_conv3d_indice_data(origin_indices, spatial_shape, ks, st, pd)
        N_out = unique_indices_offset.shape[0] - 1  # forward conv output points = inverse conv input points

        # 3. Random inverse-conv inputs
        inv_features = torch.rand(N_out, out_channel, dtype=torch.float32) * 2 - 1
        inv_weight = torch.rand(*ks, out_channel, in_channel, dtype=torch.float32) * 2 - 1
        grad_out = torch.rand(N_in, in_channel, dtype=torch.float32) * 2 - 1

        indice_data = IndiceData(
            spatial_shape,
            origin_indices.npu(),
            unique_indices_offset.npu(),
            sorted_idx.npu(),
            outidx_pair.npu(),
            False,
        )
        sparse_input_features = inv_features.npu().requires_grad_(True)
        sparse_input = SparseConvTensor(sparse_input_features, output_indices.npu(), out_spatial_shape, batch_size)
        sparse_input.indice_dict["inverse_key"] = indice_data

        inv_conv = SparseInverseConv3d(
            out_channel,
            in_channel,
            ks,
            stride=st,
            padding=pd,
            bias=False,
            indice_key="inverse_key",
        ).npu()
        inv_conv.weight.data = inv_weight.npu()

        npu_out = inv_conv(sparse_input).features
        npu_out.backward(grad_out.npu())

        # 4. CPU reference: full SparseInverseConv3d forward and backward
        cpu_img2col_out = cpu_img2col(
            inv_features,
            unique_indices_offset,
            sorted_idx,
            N_in,
            out_channel,
            kernel_product,
        )
        cpu_out = cpu_img2col_out @ inv_weight.reshape(-1, in_channel)
        cpu_feature_grad, cpu_weight_grad = cpu_sparse_inverse_conv3d_backward(
            grad_out,
            inv_features,
            inv_weight,
            unique_indices_offset,
            sorted_idx,
            N_in,
            kernel_product,
        )

        # Compare the public SparseInverseConv3d API rather than its current internal decomposition.
        self.assertRtolEqual(npu_out.detach().cpu(), cpu_out, 1e-3, 1e-3)
        self.assertRtolEqual(sparse_input_features.grad.cpu(), cpu_feature_grad, 1e-3, 1e-3)
        self.assertRtolEqual(inv_conv.weight.grad.cpu(), cpu_weight_grad, 1e-3, 1e-3)

    def _run_channel_guard_case(self, in_channel):
        out_channel = 16
        features = torch.rand(1, in_channel, dtype=torch.float32)
        origin_indices = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).npu()
        unique_indices_offset = torch.tensor([0, 1], dtype=torch.int32).npu()
        sorted_idx = torch.tensor([0], dtype=torch.int32).npu()
        outidx_pair = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).npu()
        sparse_input = SparseConvTensor(
            features.npu(),
            origin_indices,
            [1, 1, 1],
            batch_size=1,
        )
        sparse_input.indice_dict["inverse_key"] = IndiceData(
            [1, 1, 1],
            origin_indices,
            unique_indices_offset,
            sorted_idx,
            outidx_pair,
            False,
        )
        inv_conv = SparseInverseConv3d(
            in_channel,
            out_channel,
            kernel_size=1,
            bias=False,
            indice_key="inverse_key",
        ).npu()

        with self.assertRaises(RuntimeError):
            inv_conv(sparse_input)

    def test_k1_stride1_ch16_32(self):
        # 1x1x1 kernel covers the degenerate indice mapping path.
        self._run_case(
            spatial_shape=[12, 10, 8],
            feature_num=80,
            batch_size=2,
            in_channel=16,
            out_channel=32,
            kernel_size=1,
            stride=1,
            padding=0,
            seed=11,
        )

    def test_rect_kernel_mixed_stride_ch32_64(self):
        # Non-cubic kernel with mixed stride/padding improves indice generality coverage.
        self._run_case(
            spatial_shape=[18, 16, 14],
            feature_num=120,
            batch_size=3,
            in_channel=32,
            out_channel=64,
            kernel_size=[1, 3, 3],
            stride=[1, 2, 2],
            padding=[0, 1, 1],
            seed=22,
        )

    def test_model_case_bev_low_z_k113(self):
        # Low-Z BEV-style model case with a pillar-like [1, 1, 3] kernel.
        self._run_case(
            spatial_shape=[180, 180, 5],
            feature_num=1200,
            batch_size=2,
            in_channel=128,
            out_channel=128,
            kernel_size=[1, 1, 3],
            stride=[1, 1, 2],
            padding=[0, 0, 1],
            seed=44001,
        )

    def test_model_case_stride2_ch64_128(self):
        # Model-like stride-2 case with smaller 64-channel output coverage.
        self._run_case(
            spatial_shape=[120, 120, 20],
            feature_num=1000,
            batch_size=2,
            in_channel=64,
            out_channel=128,
            kernel_size=3,
            stride=2,
            padding=1,
            seed=44002,
        )

    def test_model_case0(self):
        # Model-size case from the original SparseInverseConv3d coverage.
        self._run_case(
            spatial_shape=[80, 160, 160],
            feature_num=1529,
            batch_size=4,
            in_channel=192,
            out_channel=256,
            kernel_size=3,
            stride=2,
            padding=0,
            seed=50051,
        )

    def test_reject_channel_not_divisible_by_8(self):
        self._run_channel_guard_case(in_channel=10)

    def test_reject_channel_larger_than_1024(self):
        self._run_channel_guard_case(in_channel=1032)


if __name__ == "__main__":
    run_tests()
