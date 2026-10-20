import os
import numpy as np
import torch
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving

DEVICE = f'npu:{os.environ.get("NPU_TEST_DEVICE", "0")}'


def fp16_clamp(x, min_val=None, max_val=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min_val, max_val).half()

    return x.clamp(min_val, max_val)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    # 初始化，避免 pylint used-before-assignment
    enclosed_lt = None
    enclosed_rb = None

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious

    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min_val=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area

    return gious


def gen_inputs_random_step_mn(shape_x, shape_y, dtype):
    """两组框高概率相交。"""
    assert shape_x[-1] == 4 and shape_y[-1] == 4, "边界框形状最后一维必须为 4"

    base_shape_x = shape_x[:-1]
    base_shape_y = shape_y[:-1]
    common_batch_shape = np.broadcast_shapes(base_shape_x[:-1], base_shape_y[:-1])

    center_x = np.random.rand(*common_batch_shape, 1) * 300 + 100
    center_y = np.random.rand(*common_batch_shape, 1) * 300 + 100

    # ==================== 生成 X_TENSOR ====================
    cx_x = center_x + (np.random.rand(*base_shape_x, 1) - 0.5) * 100
    cy_x = center_y + (np.random.rand(*base_shape_x, 1) - 0.5) * 100
    w_x = np.random.rand(*base_shape_x, 1) * 120 + 80
    h_x = np.random.rand(*base_shape_x, 1) * 120 + 80

    x1_x = cx_x - w_x / 2
    y1_x = cy_x - h_x / 2
    x2_x = cx_x + w_x / 2
    y2_x = cy_x + h_x / 2
    x_np = np.concatenate([x1_x, y1_x, x2_x, y2_x], axis=-1)

    # ==================== 生成 Y_TENSOR ====================
    cx_y = center_x + (np.random.rand(*base_shape_y, 1) - 0.5) * 100
    cy_y = center_y + (np.random.rand(*base_shape_y, 1) - 0.5) * 100
    w_y = np.random.rand(*base_shape_y, 1) * 120 + 80
    h_y = np.random.rand(*base_shape_y, 1) * 120 + 80

    x1_y = cx_y - w_y / 2
    y1_y = cy_y - h_y / 2
    x2_y = cx_y + w_y / 2
    y2_y = cy_y + h_y / 2
    y_np = np.concatenate([x1_y, y1_y, x2_y, y2_y], axis=-1)

    x_np[..., 2:] = np.maximum(x_np[..., 2:], x_np[..., :2] + 5)
    y_np[..., 2:] = np.maximum(y_np[..., 2:], y_np[..., :2] + 5)
    x_tensor = torch.from_numpy(x_np.astype(dtype))
    y_tensor = torch.from_numpy(y_np.astype(dtype))

    return x_tensor.contiguous(), y_tensor.contiguous()


def gen_mmdet_outputs(bboxes1, bboxes2, mode='iou', is_aligned=True):
    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


def gen_driving_outputs(bboxes1, bboxes2, mode='iou', is_aligned=True):
    # 解开关键字参数，完美兼容 PyTorch 2.7+ 底层 C++ 限制
    return mx_driving.npu_bbox_overlaps(bboxes1, bboxes2, mode, is_aligned, 1e-6)


class TestBboxOverlaps(TestCase):
    def setUp(self):
        super().setUp()
        torch.npu.set_device(DEVICE)

    def tearDown(self):
        # 及时清空显存缓存，防止合并后连续高强度测试导致 OOM
        torch.npu.empty_cache()
        super().tearDown()

    def test_npu_bbox_overlaps_precision(self):
        precision_cases = [
            (1000, 1000, 'iou', True, "ious_131", True),
            (16, 16, 'iou', True, "ious_16", True),
            (16, 16, 'giou', True, "gious", False),
            (2, 3, 'giou', False, "not_aligned", False),
        ]

        for M, N, mode, is_aligned, tag, check_tmp in precision_cases:
            with self.subTest(case=tag, mode=mode, aligned=is_aligned):
                # 1. 生成并推送数据到 NPU
                bboxes1, bboxes2 = gen_inputs_random_step_mn([M, 4], [N, 4], np.float32)
                bboxes1 = bboxes1.to(DEVICE)
                bboxes2 = bboxes2.to(DEVICE)

                # 2. 计算标准黄金数据 (mmdet)
                mmdet_result = gen_mmdet_outputs(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)

                # 3. 验证主线自研算子精度
                driving_result = gen_driving_outputs(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)
                self.assertRtolEqual(mmdet_result, driving_result, 1e-3, 1e-3)

    def test_boundary_cases(self):
        """边界值测试：相同框、不相交、完全包含、边相接、近零面积、超大坐标"""
        boundary_cases = [
            # 完全相同的框 — IoU/GIoU/IoF = 1.0
            (
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                'iou',
                True,
                "identical_iou",
            ),
            (
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                'giou',
                True,
                "identical_giou",
            ),
            ([[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]], 'iof', True, "identical_iof"),
            # 完全不相交 — IoU = 0
            ([[0.0, 0.0, 10.0, 10.0]], [[20.0, 20.0, 30.0, 30.0]], 'iou', False, "disjoint_iou"),
            ([[0.0, 0.0, 10.0, 10.0]], [[20.0, 20.0, 30.0, 30.0]], 'giou', False, "disjoint_giou"),
            # 完全包含 — 小框在大框内部
            ([[0.0, 0.0, 20.0, 20.0]], [[5.0, 5.0, 10.0, 10.0]], 'iou', False, "contained_iou"),
            ([[0.0, 0.0, 20.0, 20.0]], [[5.0, 5.0, 10.0, 10.0]], 'giou', False, "contained_giou"),
            ([[0.0, 0.0, 20.0, 20.0]], [[5.0, 5.0, 10.0, 10.0]], 'iof', False, "contained_iof"),
            # 边相接 — 零重叠面积
            ([[0.0, 0.0, 10.0, 10.0]], [[10.0, 0.0, 20.0, 10.0]], 'iou', False, "edge_touching_iou"),
            ([[0.0, 0.0, 10.0, 10.0]], [[10.0, 0.0, 20.0, 10.0]], 'giou', False, "edge_touching_giou"),
            # 近零面积框
            ([[0.0, 0.0, 1e-5, 1e-5]], [[0.0, 0.0, 1e-5, 1e-5]], 'iou', True, "near_zero_area_iou"),
            # 超大坐标
            ([[0.0, 0.0, 10000.0, 10000.0]], [[5000.0, 5000.0, 15000.0, 15000.0]], 'iou', False, "large_coord_iou"),
            ([[0.0, 0.0, 10000.0, 10000.0]], [[5000.0, 5000.0, 15000.0, 15000.0]], 'giou', False, "large_coord_giou"),
        ]

        for b1_data, b2_data, mode, is_aligned, tag in boundary_cases:
            with self.subTest(case=tag, mode=mode, aligned=is_aligned):
                bboxes1 = torch.tensor(b1_data, dtype=torch.float32).to(DEVICE)
                bboxes2 = torch.tensor(b2_data, dtype=torch.float32).to(DEVICE)

                mmdet_result = gen_mmdet_outputs(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)
                driving_result = gen_driving_outputs(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)
                self.assertRtolEqual(mmdet_result, driving_result, 1e-3, 1e-3)

    def test_boundary_expected_values(self):
        """验证边界场景的预期值"""
        # 相同框 → IoU = 1.0
        bboxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32).to(DEVICE)
        bboxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32).to(DEVICE)
        res = gen_driving_outputs(bboxes1, bboxes2, mode='iou', is_aligned=True)
        self.assertRtolEqual(res, torch.ones_like(res), 1e-3, 1e-3)

        # 不相交框 → IoU = 0.0
        bboxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32).to(DEVICE)
        bboxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float32).to(DEVICE)
        res = gen_driving_outputs(bboxes1, bboxes2, mode='iou', is_aligned=True)
        self.assertRtolEqual(res, torch.zeros_like(res), 1e-3, 1e-3)

        # 相同框 → GIoU = 1.0
        bboxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32).to(DEVICE)
        bboxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32).to(DEVICE)
        res = gen_driving_outputs(bboxes1, bboxes2, mode='giou', is_aligned=True)
        self.assertRtolEqual(res, torch.ones_like(res), 1e-3, 1e-3)

    def test_npu_bbox_overlaps_Batch_size(self):
        B1, B2 = 10, 10
        M, N = 1600, 1600
        bboxes1, bboxes2 = gen_inputs_random_step_mn([M, 4], [N, 4], np.float32)

        bboxes1 = bboxes1.reshape(B1, B2, 16, 4).to(DEVICE)
        bboxes2 = bboxes2.reshape(B1, B2, 16, 4).to(DEVICE)

        mmdet_result = gen_mmdet_outputs(bboxes1, bboxes2, mode='giou', is_aligned=True)
        driving_result = gen_driving_outputs(bboxes1, bboxes2, mode='giou', is_aligned=True)

        self.assertRtolEqual(mmdet_result, driving_result, 1e-3, 1e-3)

    def test_validation_errors(self):
        """保留：前端 if-raise 异常拦截测试"""
        test_cases = [
            ((4,), (5, 4), "iou", False, "must be at least 2-D"),
            ((2, 4), (4,), "iou", False, "must be at least 2-D"),
            ((), (2, 4), "iou", False, "must be at least 2-D"),
            ((5, 3), (5, 4), "iou", False, "The last dimension of bboxes1 must be 4"),
            ((5, 4), (5, 5), "iou", False, "The last dimension of bboxes2 must be 4"),
            ((2, 5, 4), (3, 5, 4), "iou", False, "Batch dimensions must match"),
            ((2, 3, 5, 4), (2, 4, 5, 4), "iou", False, "Batch dimensions must match"),
            ((5, 4), (6, 4), "iou", True, "must equal to N"),
            ((2, 3, 4), (2, 4, 4), "iou", True, "must equal to N"),
        ]

        for b1_shape, b2_shape, mode, is_aligned, error_msg in test_cases:
            with self.subTest(b1=b1_shape, b2=b2_shape, aligned=is_aligned):
                bboxes1 = torch.randn(b1_shape)
                bboxes2 = torch.randn(b2_shape)
                with self.assertRaisesRegex(ValueError, error_msg):
                    gen_driving_outputs(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)

    def test_empty_boxes_fast_return(self):
        """保留：空框快速返回测试"""
        empty_cases = [
            ((0, 4), (5, 4), False, (0, 5)),
            ((5, 4), (0, 4), False, (5, 0)),
            ((2, 0, 4), (2, 5, 4), False, (2, 0, 5)),
            ((3, 4, 0, 4), (3, 4, 0, 4), True, (3, 4, 0)),
        ]

        for b1_shape, b2_shape, is_aligned, expected_shape in empty_cases:
            with self.subTest(b1=b1_shape, b2=b2_shape, aligned=is_aligned):
                bboxes1 = torch.randn(b1_shape).to(DEVICE)
                bboxes2 = torch.randn(b2_shape).to(DEVICE)
                res = gen_driving_outputs(bboxes1, bboxes2, mode='iou', is_aligned=is_aligned)
                self.assertEqual(res.shape, expected_shape)
                self.assertEqual(res.numel(), 0)
                self.assertEqual(res.device.type, 'npu')


if __name__ == '__main__':
    run_tests()
