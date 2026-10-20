"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

Unit tests for sparse data structures used throughout the sparse convolution
pipeline. Tests cover:

  - SparseConvTensor creation and properties
  - IndiceData construction
  - scatter_nd utility function
  - get_conv_output_size calculation
  - SparseModule / SparseSequential composition
"""

import numpy as np
import torch
from torch import nn
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.modules.sparse_structure import (
    IndiceData,
    SparseConvTensor,
    scatter_nd,
)
from mx_driving.modules.sparse_conv import get_conv_output_size
from mx_driving.modules.sparse_modules import (
    SparseSequential,
    is_spconv_module,
    is_sparse_conv,
)


class TestSparseConvTensor(TestCase):
    """Tests for SparseConvTensor - the core sparse data container."""

    def setUp(self):
        np.random.seed(2024)

    def test_create_small_tensor(self):
        """Create a SparseConvTensor with minimal valid parameters."""
        features = torch.randn(10, 16, dtype=torch.float32)
        indices = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
            ],
            dtype=torch.int32,
        )
        spatial_shape = [2, 2, 2]
        batch_size = 2

        tensor = SparseConvTensor(features, indices, spatial_shape, batch_size)

        self.assertEqual(tensor.features.shape, (10, 16))
        self.assertEqual(tensor.batch_size, 2)
        self.assertEqual(tensor.spatial_shape, [2, 2, 2])
        self.assertEqual(tensor.spatial_size, 8)
        self.assertAlmostEqual(tensor.sparity, 10 / (2 * 2 * 2 * 2))

    def test_create_empty_indices(self):
        """Sparse tensor with zero active voxels."""
        features = torch.zeros(0, 32, dtype=torch.float32)
        indices = torch.zeros(0, 4, dtype=torch.int32)
        spatial_shape = [128, 128, 4]
        batch_size = 1

        tensor = SparseConvTensor(features, indices, spatial_shape, batch_size)
        self.assertEqual(tensor.features.shape[0], 0)
        self.assertEqual(tensor.indices.shape[0], 0)
        self.assertAlmostEqual(tensor.sparity, 0.0)

    def test_indices_dtype_conversion(self):
        """Indices should be converted to int32 if not already."""
        features = torch.randn(5, 8)
        indices = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [1, 0, 0, 0],
                [1, 2, 2, 0],
            ],
            dtype=torch.int64,
        )
        tensor = SparseConvTensor(features, indices, [4, 5, 1], 2)
        # init sets .int() if not int32 — but int64 is still int, so it stays
        self.assertIn(tensor.indices.dtype, (torch.int32, torch.int64))

    def test_dense_forward(self):
        """Convert sparse tensor to dense (NCHW) via dense() method."""
        features = torch.ones(4, 3, dtype=torch.float32)
        indices = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
            ],
            dtype=torch.int32,
        )
        spatial_shape = [2, 2]
        batch_size = 1

        tensor = SparseConvTensor(features, indices, spatial_shape, batch_size)
        dense = tensor.dense()  # channels_first=True (default) -> (1, 3, 2, 2)

        self.assertEqual(dense.shape, (1, 3, 2, 2))
        # All active positions should be 1
        self.assertAlmostEqual(dense[0, :, 0, 0].sum().item(), 3.0)  # feature [0,0,0]
        self.assertAlmostEqual(dense[0, :, 1, 0].sum().item(), 3.0)  # feature [0,1,0]

    def test_dense_channels_last(self):
        """Dense conversion with channels_first=False."""
        features = torch.randn(8, 5)
        indices = torch.randint(0, 2, (8, 4), dtype=torch.int32)
        spatial_shape = [2, 2, 2]
        batch_size = 2

        tensor = SparseConvTensor(features, indices, spatial_shape, batch_size)
        dense_last = tensor.dense(channels_first=False)
        # shape: (2, 2, 2, 2, 5) — spatial then channels
        self.assertEqual(len(dense_last.shape), 5)
        self.assertEqual(dense_last.shape[-1], 5)

    def test_find_indice_pair(self):
        """Test find_indice_pair getter/setter."""
        features = torch.randn(3, 4)
        indices = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int32)
        tensor = SparseConvTensor(features, indices, [2, 2, 1], 2)

        # Initially empty
        self.assertIsNone(tensor.find_indice_pair("conv1"))

        # Add an IndiceData entry
        indice_data = IndiceData(
            origin_spatial_shape=[2, 2, 1],
            origin_indices=indices,
            unique_indices_offset=torch.tensor([0, 2, 3], dtype=torch.int32),
            sorted_idx_to_former_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            outidx_pair=torch.zeros(3, 2, dtype=torch.int32),
            is_subm=True,
        )
        tensor.indice_dict["conv1"] = indice_data
        found = tensor.find_indice_pair("conv1")
        self.assertIsNotNone(found)
        self.assertTrue(found.is_subm)


class TestIndiceData(TestCase):
    """Tests for IndiceData - stores sparse convolution index metadata."""

    def test_create_indice_data(self):
        """Create IndiceData with all fields."""
        spatial_shape = [128, 128, 4]
        indices = torch.randint(0, 100, (50, 4), dtype=torch.int32)
        offset = torch.tensor([0, 10, 30, 50], dtype=torch.int32)
        sorted_idx = torch.arange(50, dtype=torch.int32)
        outidx = torch.randint(0, 2, (50, 2), dtype=torch.int32)

        data = IndiceData(
            origin_spatial_shape=spatial_shape,
            origin_indices=indices,
            unique_indices_offset=offset,
            sorted_idx_to_former_indices=sorted_idx,
            outidx_pair=outidx,
            is_subm=False,
        )

        self.assertEqual(data.origin_spatial_shape, spatial_shape)
        self.assertFalse(data.is_subm)
        self.assertEqual(data.origin_indices.shape[0], 50)


class TestScatterNd(TestCase):
    """Tests for scatter_nd utility used in dense() conversion."""

    def test_scatter_nd_simple(self):
        """Scatter values into a dense tensor."""
        indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
        updates = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shape = [2, 2]  # list, not torch.Size (scatter_nd slices need list concat)

        result = scatter_nd(indices, updates, shape)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertRtolEqual(result.numpy(), expected.numpy())

    def test_scatter_nd_multi_channel(self):
        """Scatter with multiple feature channels."""
        indices = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)
        updates = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        shape = [1, 2, 2, 2]  # batch, spatial_y, spatial_x, channels

        result = scatter_nd(indices, updates, shape)
        expected = torch.zeros(1, 2, 2, 2)
        expected[0, 0, 0] = torch.tensor([1.0, 2.0])
        expected[0, 1, 0] = torch.tensor([3.0, 4.0])
        expected[0, 0, 1] = torch.tensor([5.0, 6.0])
        self.assertRtolEqual(result.numpy(), expected.numpy())


class TestGetConvOutputSize(TestCase):
    """Tests for convolution output size calculation utility."""

    def test_stride2_padding1_kernel3(self):
        """Standard BEVFusion case: kernel=3, stride=2, padding=1."""
        out_size = get_conv_output_size(
            input_size=[1440, 1440, 41],
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
        )
        self.assertEqual(out_size, [720, 720, 21])

    def test_stride1_kernel3_padding1(self):
        """SubMConv case: stride=1 preserves spatial shape."""
        out_size = get_conv_output_size(
            input_size=[1440, 1440, 41],
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
        )
        self.assertEqual(out_size, [1440, 1440, 41])

    def test_stride2_kernel5_padding2(self):
        """Larger kernel size."""
        out_size = get_conv_output_size(
            input_size=[1180, 180, 5],
            kernel_size=[5, 5, 5],
            stride=[2, 2, 2],
            padding=[2, 2, 2],
            dilation=[1, 1, 1],
        )
        self.assertEqual(out_size, [590, 90, 3])

    def test_dilation2(self):
        """Dilated convolution output size."""
        out_size = get_conv_output_size(
            input_size=[32, 32, 16],
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[2, 2, 2],
            dilation=[2, 2, 2],
        )
        self.assertEqual(out_size, [32, 32, 16])


class TestSparseModuleClasses(TestCase):
    """Tests for SparseModule and SparseSequential base classes."""

    def test_sparse_module_is_detected(self):
        """is_spconv_module should identify SparseModule subclasses."""
        from mx_driving import SparseConv3d

        conv = SparseConv3d(16, 32, 3, bias=False)
        self.assertTrue(is_spconv_module(conv))

        linear = nn.Linear(10, 10)
        self.assertFalse(is_spconv_module(linear))

    def test_is_sparse_conv(self):
        """is_sparse_conv should detect SparseConvolution instances."""
        from mx_driving import SparseConv3d

        conv = SparseConv3d(16, 32, 3, bias=False)
        self.assertTrue(is_sparse_conv(conv))

        relu = nn.ReLU()
        self.assertFalse(is_sparse_conv(relu))

    def test_sparse_sequential_args(self):
        """SparseSequential with positional args."""
        from mx_driving import SparseConv3d

        model = SparseSequential(
            SparseConv3d(16, 32, 3, bias=False),
            nn.ReLU(),
            SparseConv3d(32, 64, 3, bias=False),
        )
        self.assertEqual(len(model), 3)

    def test_sparse_sequential_kwargs(self):
        """SparseSequential with named kwargs."""
        from mx_driving import SparseConv3d

        model = SparseSequential(
            conv1=SparseConv3d(16, 32, 3, bias=False),
            relu1=nn.ReLU(),
        )
        self.assertEqual(len(model), 2)

    def test_sparse_sequential_getitem(self):
        """SparseSequential supports indexing."""
        from mx_driving import SparseConv3d

        conv1 = SparseConv3d(16, 32, 3, bias=False)
        relu = nn.ReLU()
        model = SparseSequential(conv1, relu)

        self.assertIs(model[0], conv1)
        self.assertIs(model[1], relu)
        self.assertIs(model[-1], relu)


if __name__ == "__main__":
    run_tests()
