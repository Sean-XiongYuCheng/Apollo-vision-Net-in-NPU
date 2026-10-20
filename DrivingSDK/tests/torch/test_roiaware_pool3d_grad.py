import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests


class TestRoIAwarePool3dGrad(TestCase):
    @golden_data_cache(__file__)
    def roiaware_pool3d_grad_cpu(self, pts_idx_of_voxels, argmax, grad_out, npoints, pool_method):
        channels = grad_out.shape[-1]
        grad_in = torch.zeros((npoints, channels)).type_as(grad_out)

        # cast
        dtype = grad_out.dtype
        if dtype == torch.float16:
            grad_out_cast = grad_out.type(torch.float32)
            grad_in_cast = grad_in.type(torch.float32)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in

        # Compute
        if pool_method == 0:
            self.roiaware_maxpool3d_grad_cpu(argmax, grad_out_cast, grad_in_cast)

        elif pool_method == 1:
            self.roiaware_avgpool3d_grad_cpu(pts_idx_of_voxels, grad_out_cast, grad_in_cast)

        # cast
        if dtype == torch.float16:
            grad_out_cast = grad_out_cast.type(torch.float16)
            grad_in_cast = grad_in_cast.type(torch.float16)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in

        return grad_in_cast

    def roiaware_pool3d_grad_npu(self, pts_idx_of_voxels, argmax, grad_out, npoints, pool_method):
        class MockCtx:
            def __init__(self, pts_idx_of_voxels, argmax, mode, num_pts):
                self.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, mode, num_pts)

        ctx = MockCtx(pts_idx_of_voxels, argmax, pool_method, npoints)
        from mx_driving.ops.roiaware_pool3d import RoIAwarePool3dFunction

        _, _, grad_in, _, _, _ = RoIAwarePool3dFunction.backward(ctx, grad_out)
        return grad_in

    def roiaware_maxpool3d_grad_cpu(self, argmax, grad_out, grad_in):
        channels = grad_out.shape[-1]
        argmax_flat = argmax.reshape(-1, channels)
        grad_out_flat = grad_out.reshape(-1, channels)
        channel_indices = torch.arange(channels).unsqueeze(0)

        # Keep temporary index tensors bounded for large random cases while
        # preserving accumulation when multiple voxels select the same point.
        chunk_size = 1024
        for start in range(0, argmax_flat.shape[0], chunk_size):
            end = min(start + chunk_size, argmax_flat.shape[0])
            point_indices = argmax_flat[start:end]
            valid = point_indices >= 0
            if not valid.any():
                continue
            chunk_channels = channel_indices.expand(end - start, -1)
            grad_in.index_put_(
                (point_indices[valid].long(), chunk_channels[valid]),
                grad_out_flat[start:end][valid],
                accumulate=True,
            )

    def roiaware_avgpool3d_grad_cpu(self, pts_idx_of_voxels, grad_out, grad_in):
        boxes_num, out_x, out_y, out_z, channels = grad_out.shape

        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        total_pts = int(pts_idx_of_voxels[b, ox, oy, oz, 0])
                        for i in range(1, total_pts + 1):
                            pts_idx = int(pts_idx_of_voxels[b, ox, oy, oz, i])
                            grad_in[pts_idx, :] += grad_out[b, ox, oy, oz, :] / max(total_pts, 1.0)

    def gen_forward_input_data(self, pts_idx_of_voxels_shape, channels, npoints, dtype):
        boxes_num, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels_shape
        rng = np.random.default_rng(0)
        roi_centers = rng.uniform(-2, 2, (boxes_num, 3))
        roi_sizes = rng.uniform(4, 8, (boxes_num, 3))
        roi_angles = rng.uniform(-np.pi, np.pi, (boxes_num, 1))
        rois = np.concatenate((roi_centers, roi_sizes, roi_angles), axis=1).astype(dtype)
        pts = rng.uniform(-4, 6, (npoints, 3)).astype(dtype)
        pts_feature = rng.uniform(-5, 5, (npoints, channels)).astype(dtype)
        grad_out = torch.from_numpy(rng.uniform(-5, 5, (boxes_num, out_x, out_y, out_z, channels)).astype(dtype))

        class MockForwardCtx:
            roiaware_pool3d_for_backward: tuple

        from mx_driving.ops.roiaware_pool3d import RoIAwarePool3dFunction

        ctx = MockForwardCtx()
        with torch.no_grad():
            RoIAwarePool3dFunction.forward(
                ctx,
                torch.from_numpy(rois).npu(),
                torch.from_numpy(pts).npu(),
                torch.from_numpy(pts_feature).npu(),
                (out_x, out_y, out_z),
                max_pts_per_voxel,
                0,
            )
        pts_idx_of_voxels, argmax, _, _ = ctx.roiaware_pool3d_for_backward
        return argmax.cpu(), grad_out, pts_idx_of_voxels.cpu()

    def one_case(self, boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype):
        out_x, out_y, out_z = out_size
        pts_idx_of_voxels_shape = (boxes_num, out_x, out_y, out_z, max_pts_per_voxel)
        argmax, grad_out, pts_idx_of_voxels = self.gen_forward_input_data(
            pts_idx_of_voxels_shape, channels, npoints, dtype
        )

        golden_grad_in = np.zeros((npoints, channels)).astype(dtype)
        golden_grad_in = torch.from_numpy(golden_grad_in)

        golden_grad_in = self.roiaware_pool3d_grad_cpu(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)

        grad_in = np.zeros((npoints, channels)).astype(dtype)
        grad_in = torch.from_numpy(grad_in)
        grad_out = grad_out.npu()
        argmax = argmax.npu()
        grad_in = grad_in.npu()
        pts_idx_of_voxels = pts_idx_of_voxels.npu()

        grad_in = self.roiaware_pool3d_grad_npu(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)

        self.assertRtolEqual(grad_in, golden_grad_in)

    def test_roiaware_pool3d_grad(self):
        out_size = (14, 14, 14)
        self.one_case(1, out_size, 256, 128, 128, 0, "float32")
        self.one_case(1, out_size, 256, 128, 128, 1, "float32")
        self.one_case(1, out_size, 256, 128, 128, 0, "float16")
        self.one_case(1, out_size, 256, 128, 128, 1, "float16")

    def test_roiaware_maxpool3d_grad_large_unaligned_channels(self):
        # Exercise the legacy reduction path beyond its first 512 channels.
        self.one_case(5, (5, 5, 5), 516, 8, 8, 0, "float32")

    def test_roiaware_gradout_empty(self):
        pts_idx_of_voxels_shape = (1, 14, 14, 14, 128)
        argmax, grad_out, pts_idx_of_voxels = self.gen_forward_input_data(pts_idx_of_voxels_shape, 256, 128, "float32")
        grad_out = torch.tensor([]).npu()
        argmax = argmax.npu()
        pts_idx_of_voxels = pts_idx_of_voxels.npu()
        with self.assertRaises(Exception) as ctx:
            self.roiaware_pool3d_grad_npu(pts_idx_of_voxels, argmax, grad_out, 128, 0)
        self.assertEqual(str(ctx.exception), "Error! Input Tensor can not be a empty Tensor.\n")


if __name__ == "__main__":
    torch.npu.conv.allow_hf32 = False
    run_tests()
