# Copyright (c) 2026, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass

import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving

DEFAULT_RANDOM_SEED = 20260702


@dataclass(frozen=True)
class RoiawareCase:
    name: str
    boxes_num: int
    out_size: tuple
    channels: int
    npoints: int
    max_pts_per_voxel: int
    mode: int
    dtype: np.dtype


RANDOM_CASES = [
    RoiawareCase("max_float32_common", 10, (4, 4, 4), 256, 128, 128, 0, np.float32),
    RoiawareCase("avg_float32_common", 10, (4, 4, 4), 256, 128, 128, 1, np.float32),
    RoiawareCase("max_float16_channels_512", 20, (4, 4, 4), 512, 256, 128, 0, np.float16),
    RoiawareCase("avg_float16_channels_512", 20, (4, 4, 4), 512, 256, 128, 1, np.float16),
    RoiawareCase("small_voxel_grid", 6, (2, 3, 4), 64, 96, 32, 0, np.float32),
]


def lidar_to_local_coords(shift_x, shift_y, rz):
    cosa = math.cos(-rz)
    sina = math.sin(-rz)
    return shift_x * cosa + shift_y * (-sina), shift_x * sina + shift_y * cosa


def check_pt_in_box3d(point, box):
    x, y, z = point
    cx, cy, cz, x_size, y_size, z_size, rz = box
    cz += z_size / 2.0
    if abs(z - cz) > z_size / 2.0:
        return 0, 0.0, 0.0

    local_x, local_y = lidar_to_local_coords(x - cx, y - cy, rz)
    in_flag = -x_size / 2.0 < local_x < x_size / 2.0 and -y_size / 2.0 < local_y < y_size / 2.0
    return int(in_flag), local_x, local_y


@golden_data_cache(__file__)
def roiaware_pool3d_golden(rois, pts, pts_feature, out_size, max_pts_per_voxel, mode):
    num_rois = rois.shape[0]
    num_channels = pts_feature.shape[-1]
    num_pts = pts.shape[0]
    out_x, out_y, out_z = out_size

    pooled_features = np.zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=np.float32)
    pts_idx_of_voxels = np.zeros((num_rois, out_x, out_y, out_z, max_pts_per_voxel), dtype=np.int32)

    for point_idx in range(num_pts):
        for box_idx in range(num_rois):
            cur_in_flag, local_x, local_y = check_pt_in_box3d(pts[point_idx], rois[box_idx])
            if cur_in_flag <= 0:
                continue

            local_z = pts[point_idx, 2] - rois[box_idx, 2]
            x_idx = int((local_x + rois[box_idx, 3] / 2) / (rois[box_idx, 3] / out_x))
            y_idx = int((local_y + rois[box_idx, 4] / 2) / (rois[box_idx, 4] / out_y))
            z_idx = int(local_z / (rois[box_idx, 5] / out_z))
            x_idx = min(max(x_idx, 0), out_x - 1)
            y_idx = min(max(y_idx, 0), out_y - 1)
            z_idx = min(max(z_idx, 0), out_z - 1)

            cnt = pts_idx_of_voxels[box_idx, x_idx, y_idx, z_idx, 0]
            if cnt < max_pts_per_voxel - 1:
                pts_idx_of_voxels[box_idx, x_idx, y_idx, z_idx, cnt + 1] = point_idx
                pts_idx_of_voxels[box_idx, x_idx, y_idx, z_idx, 0] += 1

    for box_idx in range(num_rois):
        for x_idx in range(out_x):
            for y_idx in range(out_y):
                for z_idx in range(out_z):
                    total_pts = pts_idx_of_voxels[box_idx, x_idx, y_idx, z_idx, 0]
                    if total_pts == 0:
                        continue
                    point_indices = pts_idx_of_voxels[box_idx, x_idx, y_idx, z_idx, 1 : total_pts + 1]
                    voxel_features = pts_feature[point_indices]
                    if mode == 0:
                        pooled_features[box_idx, x_idx, y_idx, z_idx] = voxel_features.max(axis=0)
                    else:
                        pooled_features[box_idx, x_idx, y_idx, z_idx] = voxel_features.mean(axis=0)

    return pooled_features


@golden_data_cache(__file__)
def gen_random_inputs(case, seed=DEFAULT_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    xyz_center = rng.uniform(-1, 1, size=(case.boxes_num, 3)).astype(case.dtype)
    xyz_size = np.tile(rng.uniform(1, 50, size=(1, 3)), (case.boxes_num, 1)).astype(case.dtype)
    angle = np.radians(rng.integers(0, 360, size=(case.boxes_num, 1))).astype(case.dtype)
    rois = np.concatenate((xyz_center, xyz_size, angle), axis=1)
    pts = rng.uniform(-2, 4, size=(case.npoints, 3)).astype(case.dtype)
    pts_feature = rng.uniform(-1, 1, size=(case.npoints, case.channels)).astype(case.dtype)
    return rois, pts, pts_feature


class TestRoiawarePool3d(TestCase):
    def assert_case(self, rois, pts, pts_feature, out_size, max_pts_per_voxel, mode):
        golden_dtype = np.float16 if pts_feature.dtype == np.float16 else np.float32
        golden_out_size = (out_size, out_size, out_size) if isinstance(out_size, int) else out_size
        golden = roiaware_pool3d_golden(
            rois.astype(np.float32),
            pts.astype(np.float32),
            pts_feature.astype(np.float32),
            golden_out_size,
            max_pts_per_voxel,
            mode,
        ).astype(golden_dtype)

        rois_npu = torch.tensor(rois).npu()
        pts_npu = torch.tensor(pts).npu()
        pts_feature_npu = torch.tensor(pts_feature).npu()
        out = mx_driving.roiaware_pool3d(rois_npu, pts_npu, pts_feature_npu, out_size, max_pts_per_voxel, mode)

        golden_tensor = torch.tensor(golden)
        self.assertRtolEqual(golden_tensor, out.detach().cpu())

    def test_roiaware_pool3d_random(self):
        for case in RANDOM_CASES:
            with self.subTest(case=case.name):
                rois, pts, pts_feature = gen_random_inputs(case)
                self.assert_case(rois, pts, pts_feature, case.out_size, case.max_pts_per_voxel, case.mode)

    def test_roiaware_pool3d_truncates_points_per_voxel(self):
        rois = np.array([[0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0]], dtype=np.float32)
        pts = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.1, 0.2],
                [0.3, 0.1, 0.3],
                [8.0, 8.0, 8.0],
            ],
            dtype=np.float32,
        )
        pts_feature = np.array([[1.0, 1.0], [5.0, 2.0], [9.0, 9.0], [7.0, 7.0]], dtype=np.float32)
        self.assert_case(rois, pts, pts_feature, (1, 1, 1), 3, 0)
        self.assert_case(rois, pts, pts_feature, (1, 1, 1), 3, 1)

    def test_roiaware_pool3d_empty_voxels(self):
        rois = np.array([[10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        pts_feature = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.assert_case(rois, pts, pts_feature, (2, 2, 2), 4, 0)
        self.assert_case(rois, pts, pts_feature, (2, 2, 2), 4, 1)

    def test_out_size_is_int(self):
        case = RoiawareCase("int_out_size", 2, (2, 2, 2), 8, 16, 8, 0, np.float32)
        rois, pts, pts_feature = gen_random_inputs(case)
        self.assert_case(rois, pts, pts_feature, 2, case.max_pts_per_voxel, case.mode)

    def test_out_size_eq_zero(self):
        rois = torch.empty((1, 7))
        pts = torch.empty((1, 3))
        pts_feature = torch.empty((1, 1))

        invalid_out_sizes = (0, (0, 1, 2), (1, 0, 2), (1, 2, 0))
        for out_size in invalid_out_sizes:
            with self.subTest(out_size=out_size):
                with self.assertRaisesRegex(Exception, "Error! out_size can not be 0."):
                    mx_driving.roiaware_pool3d(rois, pts, pts_feature, out_size, 1, 0)

    def test_max_pts_per_voxel_eq_zero(self):
        rois = torch.empty((1, 7))
        pts = torch.empty((1, 3))
        pts_feature = torch.empty((1, 1))

        with self.assertRaisesRegex(Exception, "Error! max_pts_per_voxel can not be 0."):
            mx_driving.roiaware_pool3d(rois, pts, pts_feature, (1, 1, 1), 0, 0)

    def test_out_size_attr_error(self):
        rois = torch.empty((1, 7))
        pts = torch.empty((1, 3))
        pts_feature = torch.empty((1, 1))

        for out_size in ((1, 2), (1.0, 2.0, 3.0)):
            with self.subTest(out_size=out_size):
                with self.assertRaisesRegex(Exception, "outsize attr Error!"):
                    mx_driving.roiaware_pool3d(rois, pts, pts_feature, out_size, 1, 0)


if __name__ == "__main__":
    run_tests()
