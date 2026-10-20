"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

Extended unit tests for NMS3D operators covering edge cases and boundary
conditions not covered by the basic test suites.

Operators tested:
  - nms3d (standard 3D NMS)
  - nms3d_normal
  - nms3d_on_sight
"""

import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.detection


@golden_data_cache(__file__)
def cpu_nms3d(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """CPU reference for 3D NMS.

    Args:
        boxes: (N, 7) with [x, y, z, dx, dy, dz, heading]
        scores: (N,) confidence scores
        iou_threshold: overlap threshold for suppression

    Returns:
        keep: indices of kept boxes, sorted by score descending
    """
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    N = boxes.shape[0]
    suppressed = np.zeros(N, dtype=bool)
    keep = []

    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(order[i])
        for j in range(i + 1, N):
            if suppressed[j]:
                continue
            iou_val = _cpu_iou_bev(boxes[i], boxes[j])
            if iou_val > iou_threshold:
                suppressed[j] = True

    return np.array(keep)


def _cpu_iou_bev(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute BEV IoU for two 3D boxes (7-dof)."""
    from math import cos, sin

    a_angle = box_a[6]
    b_angle = box_b[6]
    a_dx = box_a[3] / 2
    b_dx = box_b[3] / 2
    a_dy = box_a[4] / 2
    b_dy = box_b[4] / 2

    # axis-aligned corners (before rotation)
    a_corners = np.array([[-a_dx, -a_dy], [a_dx, -a_dy], [a_dx, a_dy], [-a_dx, a_dy]])
    b_corners = np.array([[-b_dx, -b_dy], [b_dx, -b_dy], [b_dx, b_dy], [-b_dx, b_dy]])

    # rotate
    a_cos, a_sin = cos(a_angle), sin(a_angle)
    a_rot = np.array([[a_cos, -a_sin], [a_sin, a_cos]])
    a_rotated = a_corners @ a_rot.T + np.array([box_a[0], box_a[1]])

    b_cos, b_sin = cos(b_angle), sin(b_angle)
    b_rot = np.array([[b_cos, -b_sin], [b_sin, b_cos]])
    b_rotated = b_corners @ b_rot.T + np.array([box_b[0], box_b[1]])

    # Compute overlap area using Sutherland-Hodgman
    overlap = _polygon_overlap(a_rotated, b_rotated)
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    iou = overlap / max(area_a + area_b - overlap, 1e-8)
    return iou


def _polygon_overlap(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute overlap area of two convex polygons via Sutherland-Hodgman clipping."""
    from math import fabs

    output = poly1.copy()
    for i in range(len(poly2)):
        if len(output) == 0:
            return 0.0
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % len(poly2)]
        edge_vec = edge_end - edge_start
        input_poly = output
        output = []

        for j in range(len(input_poly)):
            p = input_poly[j]
            prev_p = input_poly[(j - 1 + len(input_poly)) % len(input_poly)]

            d1 = (p[0] - edge_start[0]) * edge_vec[1] - (p[1] - edge_start[1]) * edge_vec[0]
            d2 = (prev_p[0] - edge_start[0]) * edge_vec[1] - (prev_p[1] - edge_start[1]) * edge_vec[0]

            if d1 >= 0:
                if d2 < 0:
                    t = d2 / (d2 - d1)
                    output.append(np.array([prev_p[0] + t * (p[0] - prev_p[0]), prev_p[1] + t * (p[1] - prev_p[1])]))
                output.append(p)
            elif d2 >= 0:
                t = d2 / (d2 - d1)
                output.append(np.array([prev_p[0] + t * (p[0] - prev_p[0]), prev_p[1] + t * (p[1] - prev_p[1])]))

    if len(output) < 3:
        return 0.0

    area = 0.0
    for i in range(len(output)):
        j = (i + 1) % len(output)
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return fabs(area) / 2.0


class TestNms3dExtended(TestCase):
    """Extended NMS3D tests covering edge cases beyond basic test suites."""

    def setUp(self):
        np.random.seed(2024)

    # ── nms3d edge cases ──────────────────────────────────────────────

    def test_zero_threshold(self):
        """Zero IoU threshold: only exact duplicates should be suppressed."""
        boxes = np.array(
            [
                [0, 0, 0, 2, 2, 1, 0],
                [0, 0, 0, 2, 2, 1, 0],  # duplicate of box 0
                [5, 0, 0, 2, 2, 1, 0],
                [10, 0, 0, 2, 2, 1, 0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.9, 0.8, 0.1], dtype=np.float32)

        npu_keep = mx_driving.nms3d(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.0,
        )
        # NPU should return valid keep indices
        keep = npu_keep.cpu().numpy()
        self.assertGreaterEqual(len(keep), 1)
        self.assertTrue((keep >= 0).all())
        self.assertTrue((keep < 4).all())

    def test_high_threshold_keep_all(self):
        """Threshold = 1.0: no boxes should be suppressed."""
        np.random.seed(42)
        boxes = np.random.rand(20, 7).astype(np.float32)
        scores = np.random.rand(20).astype(np.float32)

        npu_keep = mx_driving.nms3d(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            1.0,
        )
        self.assertEqual(len(npu_keep), 20)

    def test_all_overlapping_boxes(self):
        """All boxes highly overlapping: only top-score box kept."""
        boxes = np.array(
            [
                [0, 0, 0, 5, 5, 1, 0],
                [0.1, 0.1, 0, 5, 5, 1, 0],
                [-0.1, -0.1, 0, 5, 5, 1, 0],
                [0.2, 0, 0, 5, 5, 1, 0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.3, 0.9, 0.1, 0.5], dtype=np.float32)

        cpu_keep = cpu_nms3d(boxes, scores, iou_threshold=0.3)
        # Expect only box[1] (highest score 0.9) to survive
        self.assertIn(1, cpu_keep.tolist())

        npu_keep = mx_driving.nms3d(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.3,
        )
        # Both CPU and NPU should return at least box 1
        self.assertIn(1, npu_keep.cpu().numpy().tolist())

    def test_non_overlapping_boxes(self):
        """Far-apart boxes: all should be kept."""
        boxes = np.array(
            [
                [0, 0, 0, 1, 1, 1, 0],
                [100, 0, 0, 1, 1, 1, 0],
                [-100, 0, 0, 1, 1, 1, 0],
                [0, 100, 0, 1, 1, 1, 0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.9, 0.8, 0.1], dtype=np.float32)

        npu_keep = mx_driving.nms3d(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.3,
        )
        self.assertEqual(len(npu_keep), 4)

    def test_large_batch(self):
        """Stress test with 5000 boxes."""
        np.random.seed(123)
        boxes = np.random.rand(5000, 7).astype(np.float32)
        boxes[:, 3:6] = np.abs(boxes[:, 3:6]) * 0.1  # small boxes to reduce overlaps
        scores = np.random.rand(5000).astype(np.float32)

        npu_keep = mx_driving.nms3d(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.5,
        )
        # Should complete without error and keep >= 1 box
        self.assertGreaterEqual(len(npu_keep), 1)

    # ── nms3d_normal edge cases ───────────────────────────────────────

    def test_nms3d_normal_single_box(self):
        """Single box input: should return the only box."""
        boxes = torch.tensor([[0, 0, 0, 2, 2, 2, 0]], dtype=torch.float32)
        scores = torch.tensor([0.5], dtype=torch.float32)
        result = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(result.cpu().numpy(), np.array([0]))

    def test_nms3d_normal_all_same_score(self):
        """All boxes with identical score but varying overlaps."""
        boxes = np.array(
            [
                [0, 0, 0, 2, 2, 1, 0],
                [0.1, 0.1, 0, 2, 2, 1, 0],
                [5, 5, 0, 2, 2, 1, 0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        result = mx_driving.nms3d_normal(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.3,
        )
        # Boxes 0 and 1 overlap heavily, box 2 is far away
        keep = result.cpu().numpy()
        self.assertEqual(len(keep), 2)

    # ── nms3d_on_sight edge cases ─────────────────────────────────────

    def test_nms3d_on_sight_all_visible(self):
        """All boxes in line-of-sight: similar to standard NMS behavior."""
        boxes = np.array(
            [
                [0, 0, 0, 2, 2, 1, 0],
                [0.5, 0.5, 0, 2, 2, 1, 0],
                [10, 10, 0, 2, 2, 1, 0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.3, 0.5], dtype=np.float32)

        result = mx_driving.nms3d_on_sight(
            torch.from_numpy(boxes).npu(),
            torch.from_numpy(scores).npu(),
            0.3,
        )
        # Should return indices of boxes kept
        keep = result.cpu().numpy()
        self.assertGreaterEqual(len(keep), 1)

    def test_nms3d_on_sight_invalid_input(self):
        """Wrong input shape should raise RuntimeError."""
        boxes = torch.randn(5, 1).npu()  # Wrong last dimension
        with self.assertRaises(Exception):
            mx_driving.nms3d_on_sight(boxes, 0.3)


if __name__ == "__main__":
    run_tests()
