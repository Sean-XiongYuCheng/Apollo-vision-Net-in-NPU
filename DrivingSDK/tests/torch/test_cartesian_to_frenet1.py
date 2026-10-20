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
    data = [pt, poly_line]
    return data


@golden_data_cache(__file__)
def gen_data_multi_d(n, bs, k1, k2, dim):
    pt = np.random.rand(n, bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(n, bs, k2, dim).astype(np.float32)
    data = [pt, poly_line]
    return data


class TestCartesianToFrenet1(TestCase):
    def golden_op(self, pt, poly_line):
        """
        Args:
            pt: (*shape, k1, 2)
            poly_line: (*shape, k2, 2)
            *shape must can be broadcasted
        Returns:
            min_idx: (*shape, k1)
            back_index: (*shape, k1)
        """

        dist_vec = torch.unsqueeze(pt, dim=-2) - torch.unsqueeze(poly_line, dim=-3)  # [s, p, t, 2]
        dist_all = torch.norm(dist_vec, dim=-1)  # (*shape, k1, k2) [s, p, t]
        min_idx = torch.argmin(dist_all, dim=-1).int()  # [s, p]
        zero = torch.tensor(0, dtype=min_idx.dtype, device=min_idx.device)
        one = torch.tensor(1, dtype=min_idx.dtype, device=min_idx.device)

        back_index = torch.where(min_idx - one < zero, zero, min_idx - one)
        return min_idx, back_index

    def test_cartesian_to_frenet1_should_return_right_value_when_shape_is_all_one(self):
        bs = 1
        k1 = 1
        k2 = 2
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_shape_is_align_to_8(self):
        bs = 32
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_shape_is_not_align(self):
        bs = 110
        k1 = 30
        k2 = 15
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_bs_is_5000(self):
        bs = 5000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_bs_greater_than_5000(self):
        bs = 10000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_shape_maxed(self):
        bs = 10000
        k1 = 256
        k2 = 256
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_k2_is_larger_than_k1(self):
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_return_right_value_when_shape_has_more_than_three_dimensions(self):
        n = 2
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line = gen_data_multi_d(n, bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx_expected, back_index_expected = self.golden_op(pt, poly_line)

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_index = mx_driving._C.cartesian_to_frenet1(dist_vec)

        # Flatten expected to (B*N, K1) to match the reshaped kernel inputs.
        min_idx_expected = min_idx_expected.reshape(-1, k1)
        back_index_expected = back_index_expected.reshape(-1, k1)

        min_idx_expected = min_idx_expected.cpu().numpy()
        back_index_expected = back_index_expected.cpu().numpy()

        min_idx = min_idx.cpu().numpy()
        back_index = back_index.cpu().numpy()

        self.assertRtolEqual(min_idx_expected, min_idx)
        self.assertRtolEqual(back_index_expected, back_index)

    def test_cartesian_to_frenet1_should_raise_error_value_when_k2_is_1(self):
        bs = 5000
        k1 = 20
        k2 = 1
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if pt_shape_ori[:-2] != poly_line_shape_ori[:-2]:
            raise ValueError("Point and poly_line must share the same batch size.")

        if len(pt_shape_ori) > 3:
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        with self.assertRaisesRegex(Exception, "Number of points in polyline must be greater than 1."):
            mx_driving._C.cartesian_to_frenet1(dist_vec)


if __name__ == "__main__":
    run_tests()
