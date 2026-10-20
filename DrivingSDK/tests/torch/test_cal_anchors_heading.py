import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


class TestCalAnchorsHeading(TestCase):
    @golden_data_cache(__file__)
    def cal_anchors_heading_cpu(self, anchors, origin_pos=None):
        if origin_pos is None:
            input_add_start = torch.cat((torch.zeros_like(anchors[:, :, 0:1, :]), anchors), dim=-2)
        else:
            input_add_start = torch.cat(
                (origin_pos.unsqueeze(1).unsqueeze(1).repeat(1, anchors.shape[1], 1, 1), anchors), dim=-2
            )

        xy_diff = input_add_start[:, :, 1:, :] - input_add_start[:, :, :-1, :]
        heading_valid = torch.logical_or(xy_diff[..., 0] > 0.1, xy_diff[..., 1] > 0.1)
        heading = torch.atan2(xy_diff[..., 1], xy_diff[..., 0])

        for t in range(heading.shape[2]):
            heading_t = heading[:, :, t]
            heading_valid_t = heading_valid[:, :, t]
            if t == 0:
                heading_t[~heading_valid_t] = 0
            else:
                heading_t[~heading_valid_t] = heading[:, :, t - 1][~heading_valid_t]

        return heading.numpy()

    def cal_anchors_heading_npu(self, anchors, origin_pos=None):
        anchors = anchors.npu()
        origin_pos = None if origin_pos is None else origin_pos.npu()
        heading = mx_driving.cal_anchors_heading(anchors, origin_pos)
        return heading.cpu().numpy()

    @golden_data_cache(__file__)
    def gen_data(self, batch_size, anchors_num, seq_length):
        anchors = np.random.uniform(-5, 5, (batch_size, anchors_num, seq_length, 2))
        origin_pos = np.random.uniform(-5, 5, (batch_size, 2))
        return torch.from_numpy(anchors).float(), torch.from_numpy(origin_pos).float()

    def one_case(self, batch_size, anchors_num, seq_length, none_origin_pos=False):
        anchors, origin_pos = self.gen_data(batch_size, anchors_num, seq_length)
        origin_pos = origin_pos if none_origin_pos is False else None
        heading_cpu = self.cal_anchors_heading_cpu(anchors, origin_pos)
        heading_npu = self.cal_anchors_heading_npu(anchors, origin_pos)
        self.assertRtolEqual(heading_cpu, heading_npu)

    def test_cal_anchors_heading(self):
        # test min border
        self.one_case(1, 1, 1, True)
        self.one_case(1, 1, 1, False)

        # test batch_size max border
        self.one_case(2048, 256, 32, True)
        self.one_case(2048, 256, 32, False)

        # test anchor max border
        self.one_case(2, 10240, 32, True)
        self.one_case(2, 10240, 32, False)

        # test seq_legnth max border
        self.one_case(2, 256, 256, True)
        self.one_case(2, 256, 256, False)

    def test_zero_displacement_all_invalid(self):
        """All consecutive points have xy-diff <= 0.1 — every heading should be 0."""
        # seq_length=3: origin(0,0) -> (0.05,0) -> (0.10,0) -> (0.15,0)
        # All |dx| <= 0.1, |dy| = 0 → all steps INVALID
        # t=0 invalid→0; t=1 invalid→inherit(t=0)=0; t=2 invalid→inherit(t=1)=0
        anchors = torch.zeros(1, 1, 3, 2)
        anchors[0, 0, 0] = torch.tensor([0.05, 0.0])
        anchors[0, 0, 1] = torch.tensor([0.10, 0.0])
        anchors[0, 0, 2] = torch.tensor([0.15, 0.0])

        heading_cpu = self.cal_anchors_heading_cpu(anchors)
        heading_npu = self.cal_anchors_heading_npu(anchors)
        self.assertRtolEqual(heading_cpu, heading_npu)
        self.assertRtolEqual(np.zeros_like(heading_cpu), heading_npu)

    def test_zero_displacement_multi_anchor(self):
        """Multiple anchors, all with tiny displacements — all headings should be 0."""
        anchors = torch.zeros(2, 4, 5, 2)
        # Each step moves by at most 0.08 in x or y
        for t in range(1, 5):
            anchors[:, :, t, 0] = anchors[:, :, t - 1, 0] + 0.08
            anchors[:, :, t, 1] = anchors[:, :, t - 1, 1] + 0.03

        heading_cpu = self.cal_anchors_heading_cpu(anchors)
        heading_npu = self.cal_anchors_heading_npu(anchors)
        self.assertRtolEqual(heading_cpu, heading_npu)
        self.assertRtolEqual(np.zeros_like(heading_cpu), heading_npu)

    def test_mixed_valid_invalid(self):
        """Alternating valid and invalid steps — verify fallback inheritance."""
        # 1 batch, 1 anchor, seq_length=4
        # Points: origin(0,0) -> (2,0) -> (2.05,0) -> (0,2) -> (0.05,1.95)
        # step0 dx=2 → VALID, heading=atan2(0,2)=0
        # step1 dx=0.05 → INVALID, inherit step0 → 0
        # step2 dx=-2.05, dy=2 → VALID, heading=atan2(2,-2.05)
        # step3 dx=0.05, dy=-0.05 → INVALID, inherit step2
        anchors = torch.zeros(1, 1, 4, 2)
        anchors[0, 0, 0] = torch.tensor([2.0, 0.0])
        anchors[0, 0, 1] = torch.tensor([2.05, 0.0])
        anchors[0, 0, 2] = torch.tensor([0.0, 2.0])
        anchors[0, 0, 3] = torch.tensor([0.05, 1.95])

        heading_cpu = self.cal_anchors_heading_cpu(anchors)
        heading_npu = self.cal_anchors_heading_npu(anchors)
        self.assertRtolEqual(heading_cpu, heading_npu)

        # Verify that invalid steps inherited from previous valid steps
        heading_cpu_np = heading_cpu[0, 0]
        # step0 and step1 should both be ~0
        self.assertRtolEqual(heading_cpu_np[0], heading_cpu_np[1])
        # step2 and step3 should be equal
        self.assertRtolEqual(heading_cpu_np[2], heading_cpu_np[3])
        # step0 != step2 (different heading)
        self.assertGreater(
            abs(heading_cpu_np[0] - heading_cpu_np[2]), 0.1, "Expected step0 and step2 to have different headings"
        )

    def test_extreme_angles(self):
        """Test headings at quadrant boundaries to exercise atan2 edges."""
        # 1 batch, 2 anchors, seq_length=3
        # Anchor 0: near ±π boundary
        #   origin(0,0) -> (-1, 0) -> (-2, -1e-6) -> (-3, -2)
        # Anchor 1: near axis crossings
        #   origin(0,0) -> (1, 0) -> (0, 1) -> (-1, 0)
        #   step0: (1,0) → atan2(0,1)=0
        #   step1: (-1,1) → atan2(1,-1)≈2.356 (135°)
        #   step2: (-1,-1) → atan2(-1,-1)≈-2.356 (-135°)
        anchors = torch.zeros(1, 2, 3, 2)
        # Anchor 0: near ±π
        anchors[0, 0, 0] = torch.tensor([-1.0, 0.0])
        anchors[0, 0, 1] = torch.tensor([-2.0, -1e-6])
        anchors[0, 0, 2] = torch.tensor([-3.0, -2.0])
        # Anchor 1: axis crossings
        anchors[0, 1, 0] = torch.tensor([1.0, 0.0])
        anchors[0, 1, 1] = torch.tensor([0.0, 1.0])
        anchors[0, 1, 2] = torch.tensor([-1.0, 0.0])

        heading_cpu = self.cal_anchors_heading_cpu(anchors)
        heading_npu = self.cal_anchors_heading_npu(anchors)
        self.assertRtolEqual(heading_cpu, heading_npu)

    def test_all_valid_large_displacement(self):
        """All steps have large displacements — no invalid fallback needed."""
        # 1 batch, 1 anchor, 4 steps, all with |dx| or |dy| >> 0.1
        # origin(0,0) -> (1,1) -> (3,1) -> (3,4) -> (0,0)
        # step0: dx=1,dy=1 → heading=π/4
        # step1: dx=2,dy=0 → heading=0 (horizontal)
        # step2: dx=0,dy=3 → heading=π/2
        # step3: dx=-3,dy=-4 → heading≈-2.214
        anchors = torch.zeros(1, 1, 4, 2)
        anchors[0, 0, 0] = torch.tensor([1.0, 1.0])
        anchors[0, 0, 1] = torch.tensor([3.0, 1.0])
        anchors[0, 0, 2] = torch.tensor([3.0, 4.0])
        anchors[0, 0, 3] = torch.tensor([0.0, 0.0])

        heading_cpu = self.cal_anchors_heading_cpu(anchors)
        heading_npu = self.cal_anchors_heading_npu(anchors)
        self.assertRtolEqual(heading_cpu, heading_npu)

        # Verify geometric correctness: headings match known directions
        heading_cpu_np = heading_cpu[0, 0]
        self.assertTrue(
            abs(heading_cpu_np[0] - np.pi / 4) < 0.01, f"step0: expected ~π/4, got {heading_cpu_np[0]}"
        )  # 45° up-right
        self.assertTrue(abs(heading_cpu_np[1]) < 0.01, f"step1: expected ~0, got {heading_cpu_np[1]}")  # horizontal
        self.assertTrue(
            abs(heading_cpu_np[2] - np.pi / 2) < 0.01, f"step2: expected ~π/2, got {heading_cpu_np[2]}"
        )  # 90° up

    def test_with_origin_pos(self):
        """Test with explicit non-zero origin positions."""
        # 2 batches, 2 anchors, seq_length=2
        # origin_pos varies per batch but not per anchor
        anchors = torch.zeros(2, 2, 2, 2)
        anchors[0, 0, 0] = torch.tensor([1.0, 0.0])
        anchors[0, 0, 1] = torch.tensor([2.0, 0.0])
        anchors[0, 1, 0] = torch.tensor([0.0, 1.0])
        anchors[0, 1, 1] = torch.tensor([0.0, 2.0])
        anchors[1, 0, 0] = torch.tensor([-1.0, -1.0])
        anchors[1, 0, 1] = torch.tensor([-2.0, -2.0])
        anchors[1, 1, 0] = torch.tensor([1.0, -1.0])
        anchors[1, 1, 1] = torch.tensor([2.5, -2.0])
        origin_pos = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])

        heading_cpu = self.cal_anchors_heading_cpu(anchors, origin_pos)
        heading_npu = self.cal_anchors_heading_npu(anchors, origin_pos)
        self.assertRtolEqual(heading_cpu, heading_npu)


if __name__ == "__main__":
    run_tests()
