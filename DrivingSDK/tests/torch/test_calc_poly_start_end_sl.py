# pylint: disable=R0801
import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


def batch_index_select_changed_np(A, indices):
    if A.ndim == 3:
        b, n, d = A.shape
        offset = indices.astype(np.int64) + np.arange(0, b * n, n)[:, None]
        return A.reshape(b * n, d)[offset]
    if A.ndim == 4:
        b, a, n, d = A.shape
        m = indices.shape[-1]
        A = A.reshape(b * a * n, d)
        indices = indices.reshape(b * a, m)
        offset = indices.astype(np.int64) + np.arange(0, b * a * n, n)[:, None]
        return A[offset].reshape(b, a, m, d)
    return None


@golden_data_cache(__file__)
def gen_data(bs, k1, k2, dim):
    pt = np.random.rand(bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(bs, k2, dim).astype(np.float32)

    dist_vec = np.expand_dims(pt, -2) - np.expand_dims(poly_line, -3)  # (bs, k1, k2, dim)
    dist_all = np.linalg.norm(dist_vec, axis=-1)  # (bs, k1, k2)
    min_idx = np.argmin(dist_all, axis=-1).astype(np.int32)  # (bs, k1)

    back_index = np.where(min_idx - 1 < 0, 0, min_idx - 1).astype(np.int32)  # (bs, k1)
    front_point = batch_index_select_changed_np(poly_line, min_idx)
    back_point = batch_index_select_changed_np(poly_line, back_index)

    move_forward_point = np.sum((pt - front_point) * (back_point - front_point), axis=-1) > 0
    tmp_idx = np.logical_and(min_idx > 0, move_forward_point)
    tmp_idx = np.logical_or(tmp_idx, min_idx == poly_line.shape[-2] - 1)
    min_idx = np.where(tmp_idx, min_idx - 1, min_idx)

    data = [pt, poly_line, min_idx]
    return data


@golden_data_cache(__file__)
def gen_data_multi_d(n, bs, k1, k2, dim):
    pt = np.random.rand(n, bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(n, bs, k2, dim).astype(np.float32)

    dist_vec = np.expand_dims(pt, -2) - np.expand_dims(poly_line, -3)
    dist_all = np.linalg.norm(dist_vec, axis=-1)
    min_idx = np.argmin(dist_all, axis=-1).astype(np.int32)
    back_index = np.where(min_idx - 1 < 0, 0, min_idx - 1).astype(np.int32)

    front_point = batch_index_select_changed_np(poly_line, min_idx)
    back_point = batch_index_select_changed_np(poly_line, back_index)

    move_forward_point = np.sum((pt - front_point) * (back_point - front_point), axis=-1) > 0
    tmp_idx = np.logical_and(min_idx > 0, move_forward_point)
    tmp_idx = np.logical_or(tmp_idx, min_idx == poly_line.shape[-2] - 1)
    min_idx = np.where(tmp_idx, min_idx - 1, min_idx)

    data = [pt, poly_line, min_idx]
    return data


class TestCalcPolyStartEndSl(TestCase):
    def batch_index_select(self, A, indices):
        """Slice input tensor with indicies in batch-wise
        Args:
            A: (B x A x N x D) or (B x N x D)
            indices: (B x A x M) or (B x M) (select M indices, element value between 0~N-1)
        return:
            out: (B x A x M x D) or (B x M x D)
        """
        assert isinstance(A, (torch.Tensor, np.ndarray)), "Input type must be np.ndarray or torch.Tensor"
        assert len(A.shape) == 3 or len(A.shape) == 4, "Input shape must be with 3 or 4 dimensions"
        if len(A.shape) == 3:
            assert len(indices.shape) == 2 and indices.shape[:1] == A.shape[:1], "Input size mismatched"
            b, n, d = A.shape
            if isinstance(A, torch.Tensor):
                indicies_offset = indices.type(torch.int32) + torch.arange(
                    0, b * n, n, dtype=torch.int32, device=indices.device
                ).unsqueeze(1)
                output = A.reshape(b * n, d)[indicies_offset.to(dtype=torch.int64)]
            elif isinstance(A, np.ndarray):
                indicies_offset = indices + np.arange(0, b * n, n)[..., np.newaxis]
                output = A.reshape(b * n, d)[indicies_offset]
            return output
        if len(A.shape) == 4:
            assert len(indices.shape) == 3 and indices.shape[:2] == A.shape[:2], "Input size mismatched"
            b, a, n, d = A.shape
            _, _, m = indices.shape
            indices = indices.reshape(b * a, m)
            indicies_offset = indices.type(torch.int32) + torch.arange(
                0, b * a * n, n, dtype=torch.int32, device=indices.device
            ).unsqueeze(1)
            output = A.reshape(b * a * n, d)[indicies_offset.to(dtype=torch.int64)]
            return output.reshape(b, a, m, d)
        return None

    def golden_op(self, pt, poly_line, min_idx):
        """
        Args:
            pt: (*shape, k1, 2)
            poly_line: (*shape, k2, 2)
            min_idx: (*shape, k1)
            *shape must can be broadcasted
        Returns:
            poly_start: (*shape, k1, 2)
            poly_end: (*shape, k1, 2)
            sl: (*shape, k1, 2)
        """

        def _dot(pts1, pts2):
            return torch.sum((pts1 * pts2), -1)

        def _cross(pts1, pts2):
            return pts1[..., 0] * pts2[..., 1] - pts1[..., 1] * pts2[..., 0]

        def _pt_to_line(pt, pt1, pt2):
            sub_pt_pt1 = pt - pt1
            sub_pt2_pt1 = pt2 - pt1
            s = _dot(sub_pt_pt1, sub_pt2_pt1) / torch.clip(torch.norm(sub_pt2_pt1, dim=-1), 1e-5, 1e10)
            length = -_cross(sub_pt_pt1, sub_pt2_pt1) / torch.clip(torch.norm(sub_pt2_pt1, dim=-1), 1e-5, 1e10)
            return s, length

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start = self.batch_index_select(poly_line, min_idx)
        poly_end = self.batch_index_select(poly_line, min_idx + 1)
        s, length = _pt_to_line(pt, poly_start, poly_end)  # (*shape, k1)
        s = s + self.batch_index_select(s_cum.unsqueeze(-1), min_idx).squeeze(-1)  # (*shape, k1)
        return poly_start, poly_end, torch.stack((s, length), dim=-1)  # (*shape, k1, 2)

    def test_calc_poly_start_end_sl_should_return_right_value_when_shape_is_all_one(self):
        bs = 1
        k1 = 1
        k2 = 2
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_shape_is_align_to_8(self):
        bs = 32
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_shape_is_not_align(self):
        bs = 110
        k1 = 30
        k2 = 15
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_bs_is_5000(self):
        bs = 5000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_bs_greater_than_5000(self):
        bs = 10000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_shape_maxed(self):
        bs = 10000
        k1 = 256
        k2 = 256
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_k2_is_larger_than_k1(self):
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_return_right_value_when_shape_has_more_than_three_dimensions(self):
        n = 2
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line, min_idx = gen_data_multi_d(n, bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx).npu()

        # The kernel expects 3D poly_line/pt and 2D min_idx/s_cum, so flatten
        # the leading dims the same way the op wrapper does.
        pt = pt.reshape(-1, pt.shape[-2], pt.shape[-1])
        poly_line = poly_line.reshape(-1, poly_line.shape[-2], poly_line.shape[-1])
        min_idx = min_idx.reshape(-1, min_idx.shape[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line, min_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_calc_poly_start_end_sl_should_raise_error_value_when_min_idx_is_not_2d(self):
        bs = 8
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        min_idx = torch.from_numpy(min_idx.reshape(-1)).npu()

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)
        s_cum = torch.cumsum(diff_dist, dim=-1)

        with self.assertRaisesRegex(Exception, r"min_idx\.dim\(\) must be 2, but got: 1"):
            mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

    def test_calc_poly_start_end_sl_should_raise_error_value_when_poly_line_is_not_3d(self):
        bs = 8
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line, min_idx = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(np.random.rand(bs, k2).astype(np.float32)).npu()
        min_idx = torch.from_numpy(min_idx).npu()
        s_cum = torch.zeros(bs, k2, device=poly_line.device)

        with self.assertRaisesRegex(Exception, r"poly_line\.dim\(\) must be 3, but got: 2"):
            mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)


if __name__ == "__main__":
    run_tests()
