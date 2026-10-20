# pylint: disable=R0801
import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


@golden_data_cache(__file__)
def gen_data(bs, k1, k2, dim):
    pt = np.random.rand(bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(bs, k2, dim).astype(np.float32)

    dist_vec = np.expand_dims(pt, -2) - np.expand_dims(poly_line, -3)  # (bs, k1, k2, dim)
    dist_all = np.linalg.norm(dist_vec, axis=-1)  # (bs, k1, k2)
    min_idx = np.argmin(dist_all, axis=-1).astype(np.int32)  # (bs, k1)

    back_index = np.where(min_idx - 1 < 0, 0, min_idx - 1).astype(np.int32)  # (bs, k1)
    data = [pt, poly_line, min_idx, back_index]
    return data


@golden_data_cache(__file__)
def gen_data_multi_d(n, bs, k1, k2, dim):
    pt = np.random.rand(n, bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(n, bs, k2, dim).astype(np.float32)

    dist_vec = np.expand_dims(pt, -2) - np.expand_dims(poly_line, -3)
    dist_all = np.linalg.norm(dist_vec, axis=-1)
    min_idx = np.argmin(dist_all, axis=-1).astype(np.int32)

    back_index = np.where(min_idx - 1 < 0, 0, min_idx - 1).astype(np.int32)
    data = [pt, poly_line, min_idx, back_index]
    return data


class TestSelectIdxWithMask(TestCase):
    def batch_index_select_changed(self, A, indices):
        b, n, d = A.shape
        indicies_offset = indices.type(torch.int32) + torch.arange(
            0, b * n, n, dtype=torch.int32, device=indices.device
        ).unsqueeze(1)
        output = A.reshape(b * n, d)[indicies_offset.to(dtype=torch.int64)]
        return output

    def golden_op(self, pt, poly_line, min_idx, back_index):
        """
        Args:
            pt: (*shape, k1, 2)
            poly_line: (*shape, k2, 2)
            min_idx: (*shape, k1)
            back_index: (*shape, k1)
            *shape must can be broadcasted
        Returns:
            min_idx: (*shape, k1)
        """

        def _dot(pts1, pts2):
            return torch.sum((pts1 * pts2), -1)

        front_point = self.batch_index_select_changed(poly_line, min_idx)
        back_point = self.batch_index_select_changed(poly_line, back_index)
        move_forward_point = _dot(pt - front_point, back_point - front_point) > 0
        tmp_idx = torch.logical_and(min_idx > 0, move_forward_point)
        tmp_idx = torch.logical_or(tmp_idx, min_idx == poly_line.shape[-2] - 1)
        min_idx = torch.where(tmp_idx, min_idx - 1, min_idx)

        return min_idx

    def test_select_idx_with_mask_should_return_right_value_when_shape_is_all_one(self):
        bs = 1
        k1 = 1
        k2 = 2
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_shape_is_align_to_8(self):
        bs = 32
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_shape_is_not_align(self):
        bs = 110
        k1 = 30
        k2 = 15
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_bs_is_5000(self):
        bs = 5000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_bs_greater_than_5000(self):
        bs = 10000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_shape_maxed(self):
        bs = 10000
        k1 = 256
        k2 = 256
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_k2_is_larger_than_k1(self):
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()
        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_return_right_value_when_shape_has_more_than_three_dimensions(self):
        n = 2
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data_multi_d(n, bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()

        # The kernel expects 3D poly_line/pt and 2D indices, so flatten the
        # leading dims the same way the op wrapper does.
        pt = pt.reshape(-1, pt.shape[-2], pt.shape[-1])
        poly_line = poly_line.reshape(-1, poly_line.shape[-2], poly_line.shape[-1])
        min_idx = min_idx.reshape(-1, min_idx.shape[-1])
        back_idx = back_idx.reshape(-1, back_idx.shape[-1])

        min_idx_expected = self.golden_op(pt, poly_line, min_idx, back_idx)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

        min_idx_expected = min_idx_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)

    def test_select_idx_with_mask_should_raise_error_value_when_pt_and_min_idx_size_mismatch(self):
        bs = 8
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx[:, : k1 - 1]).npu()
        back_idx = torch.from_numpy(back_idx[:, : k1 - 1]).npu()

        with self.assertRaisesRegex(Exception, "The second dimension of pt must match the second dimension of min_idx"):
            mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)

    def test_select_idx_with_mask_should_raise_error_value_when_poly_line_dim_is_not_3(self):
        bs = 8
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx, back_idx = gen_data(bs, k1, k2, dim)
        poly_line = np.random.rand(bs, k2, 3).astype(np.float32)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        back_idx = torch.from_numpy(back_idx).npu()

        with self.assertRaisesRegex(Exception, "The third dimension of poly_line must be 2"):
            mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)


if __name__ == "__main__":
    run_tests()
