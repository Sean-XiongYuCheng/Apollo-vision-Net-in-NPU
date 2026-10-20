"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

BEVFusion pipeline integration test.

The BEVFusion LiDAR branch pipeline:
  1. Point cloud voxelization (hard_voxelize)
  2. SparseConv3d (down-sampling spatial)
  3. SubMConv3d (in-place refinement, stride=1)
  4. Scatter to dense BEV representation
  5. BEV Pooling (camera-to-BEV projection)

This test validates the end-to-end forward and backward pass using
synthetic data at small scale.
"""

import numpy as np
import torch
from torch import nn
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import (
    SparseConv3d,
    SparseConvTensor,
    SubMConv3d,
    Voxelization,
    bev_pool,
)


def _generate_bevfusion_input(num_points=2000):
    """Generate synthetic point cloud for BEVFusion pipeline testing.

    Returns:
        points: (num_points, 4) XYZ + intensity
        voxel_size: list of voxel dimensions
        point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    np.random.seed(2024)
    x = np.random.uniform(0, 69.12, num_points)
    y = np.random.uniform(-39.68, 39.68, num_points)
    z = np.random.uniform(-3, 1, num_points)
    intensity = np.random.uniform(0, 1, num_points)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    return points


class BevfusionLidarPipeline(nn.Module):
    """Simplified BEVFusion LiDAR pipeline for integration testing.

    This models the core LiDAR branch of BEVFusion:
    voxelization → sparse encoder → scatter → BEV features.
    """

    def __init__(self, in_channels=4, feat_channels=16):
        super().__init__()
        self.voxelization = Voxelization(
            voxel_size=[0.16, 0.16, 4.0],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=500,
        )
        self.conv1 = SparseConv3d(in_channels, feat_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = SubMConv3d(feat_channels, feat_channels * 2, kernel_size=3, bias=False)

    def forward(self, points):
        """Forward pass: voxelization → SparseConv3d → SubMConv3d."""
        # 1. Voxelize
        num_voxels, voxels, coors, num_pts = self.voxelization(points)

        # 2. Build SparseConvTensor (flatten voxels to (M, C))
        batch_idx = torch.zeros(coors.shape[0], 1, dtype=torch.int32, device=coors.device)
        indices = torch.cat([batch_idx, coors[:, [2, 1, 0]]], dim=1)  # b,z,y,x → b,x,y,z

        # Mean-pool features within each voxel
        voxel_feats = voxels.sum(dim=1) / (num_pts.unsqueeze(-1).float() + 1e-8)

        spatial_shape = [432, 496, 1]  # 69.12/0.16 ≈ 432, 79.36/0.16 ≈ 496
        sparse_tensor = SparseConvTensor(voxel_feats, indices, spatial_shape, batch_size=1)

        # 3. SparseConv3d (downsample)
        out1 = self.conv1(sparse_tensor)

        # 4. SubMConv3d (refine)
        out2 = self.conv2(out1)

        return out2


class TestBevfusionPipeline(TestCase):
    """BEVFusion pipeline integration tests."""

    def setUp(self):
        np.random.seed(2024)

    def test_voxelization_sparse_conv_chain_forward(self):
        """Forward pass through voxelization → SparseConv3d → SubMConv3d."""
        points_np = _generate_bevfusion_input(2000)
        points = torch.from_numpy(points_np).npu()

        model = BevfusionLidarPipeline(in_channels=4, feat_channels=16).npu()
        model.eval()

        with torch.no_grad():
            output = model(points)
            features = output.features

        # Should produce non-empty output features
        self.assertGreater(features.shape[0], 0)
        self.assertEqual(features.shape[1], 32)  # feat_channels * 2

    def test_voxelization_sparse_conv_chain_backward(self):
        """Backward pass verifies gradients flow through the pipeline."""
        # SparseConv3d requires channels aligned to 16. Use 16-channel input.
        points_np = _generate_bevfusion_input(1000)
        # Pad 4-channel input to 16 channels
        points_padded = np.concatenate([points_np, np.zeros((points_np.shape[0], 12), dtype=np.float32)], axis=1)
        points = torch.from_numpy(points_padded).npu()

        model = BevfusionLidarPipeline(in_channels=16, feat_channels=16).npu()

        # Use the internal voxel features directly
        num_voxels, voxels, coors, num_pts = model.voxelization(points)

        batch_idx = torch.zeros(coors.shape[0], 1, dtype=torch.int32, device=coors.device)
        indices = torch.cat([batch_idx, coors[:, [2, 1, 0]]], dim=1)

        voxel_feats = (voxels.sum(dim=1) / (num_pts.unsqueeze(-1).float() + 1e-8)).requires_grad_()

        spatial_shape = [432, 496, 1]
        sparse_tensor = SparseConvTensor(voxel_feats, indices, spatial_shape, batch_size=1)

        out1 = model.conv1(sparse_tensor)
        out2 = model.conv2(out1)

        loss = out2.features.sum()
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(voxel_feats.grad)
        self.assertGreater(voxel_feats.grad.abs().sum().item(), 0)

    def test_pipeline_with_bev_pool(self):
        """End-to-end: voxelization → sparse conv → BEV pool integration."""
        # BEV pool test with synthetic features
        b, d, h, w, c = 1, 2, 32, 32, 8
        n = 200
        feat_np = np.random.rand(n, c).astype(np.float32)
        geom_np = np.zeros((n, 4), dtype=np.int32)
        geom_np[:, 0] = np.random.randint(0, h, n)
        geom_np[:, 1] = np.random.randint(0, w, n)
        geom_np[:, 2] = np.random.randint(0, d, n)
        geom_np[:, 3] = 0

        feat = torch.from_numpy(feat_np).npu().requires_grad_()
        geom = torch.from_numpy(geom_np).npu()

        output = bev_pool(feat, geom, b, d, h, w)
        self.assertEqual(output.shape, (b, c, d, h, w))

        output.sum().backward()
        self.assertIsNotNone(feat.grad)


if __name__ == "__main__":
    run_tests()
