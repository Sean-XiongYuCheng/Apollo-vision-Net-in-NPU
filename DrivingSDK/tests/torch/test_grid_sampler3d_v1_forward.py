"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Unit tests for grid_sampler3d_v1 forward pass.
Complements the existing test_grid_sampler3d_grad_v1.py (backward test)
by validating forward output correctness across interpolation/padding modes.
"""

import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import grid_sampler3d_v1


@golden_data_cache(__file__)
def generate_grid_sampler3d_data(input_shape, grid_shape, dtype=torch.float32):
    """Generate random input and grid tensors for grid_sample testing.

    Args:
        input_shape: shape of the input feature tensor (N, C, D, H, W)
        grid_shape: shape of the grid tensor (N, D_out, H_out, W_out, 3)
        dtype: data type

    Returns:
        input_tensor, grid_tensor (both as numpy for caching)
    """
    torch.manual_seed(2024)
    input_tensor = torch.rand(input_shape, dtype=dtype)
    # grid in range [-1, 1]
    grid_tensor = 2 * torch.rand(grid_shape, dtype=dtype) - 1
    return input_tensor, grid_tensor


class TestGridSampler3dV1Forward(TestCase):
    """Forward-pass tests for grid_sampler3d_v1.

    Validates output correctness by comparing NPU result against
    PyTorch's torch.nn.functional.grid_sample as CPU reference.
    """

    def _run_forward_test(
        self, input_shape, grid_shape, mode="bilinear", padding_mode="zeros", align_corners=True, dtype=torch.float32
    ):
        """Run forward comparison between NPU operator and PyTorch reference."""
        input_tensor, grid_tensor = generate_grid_sampler3d_data(input_shape, grid_shape, dtype)

        # CPU reference: PyTorch native grid_sample
        cpu_output = torch.nn.functional.grid_sample(
            input_tensor, grid_tensor, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )

        # NPU operator
        npu_input = input_tensor.npu()
        npu_grid = grid_tensor.npu()
        npu_output = grid_sampler3d_v1(
            npu_input, npu_grid, interpolation=mode, padding=padding_mode, align=align_corners
        )

        self.assertRtolEqual(npu_output.cpu(), cpu_output)

    def test_bilinear_zeros_align_true_small(self):
        """Small input with bilinear interpolation, zero padding, aligned corners."""
        self._run_forward_test(
            input_shape=(2, 4, 4, 4, 4),
            grid_shape=(2, 2, 2, 2, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def test_bilinear_zeros_align_false(self):
        """Bilinear interpolation without aligned corners."""
        self._run_forward_test(
            input_shape=(2, 6, 8, 8, 8),
            grid_shape=(2, 1, 4, 4, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def test_bilinear_border_align_true(self):
        """Bilinear interpolation with border padding."""
        self._run_forward_test(
            input_shape=(2, 3, 6, 6, 6),
            grid_shape=(2, 3, 3, 3, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def test_bilinear_reflection_align_true(self):
        """Bilinear interpolation with reflection padding."""
        self._run_forward_test(
            input_shape=(2, 3, 6, 6, 6),
            grid_shape=(2, 1, 4, 4, 3),
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

    def test_nearest_zeros_align_true(self):
        """Nearest-neighbor interpolation with zero padding."""
        self._run_forward_test(
            input_shape=(2, 4, 4, 4, 4),
            grid_shape=(2, 2, 2, 2, 3),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )

    def test_nearest_border_align_false(self):
        """Nearest-neighbor interpolation with border padding, unaligned."""
        self._run_forward_test(
            input_shape=(2, 5, 7, 7, 7),
            grid_shape=(2, 1, 5, 5, 3),
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )

    def test_large_spatial_dims(self):
        """Larger spatial dimensions to stress-test tiling logic."""
        self._run_forward_test(
            input_shape=(4, 16, 4, 32, 64),
            grid_shape=(4, 1, 16, 32, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def test_model_like_shape(self):
        """Shape pattern similar to BEV model usage (N,C,D,H,W)."""
        self._run_forward_test(
            input_shape=(16, 64, 4, 64, 176),
            grid_shape=(16, 100, 4, 1, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


if __name__ == "__main__":
    run_tests()
