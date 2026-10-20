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
from dataclasses import dataclass

import torch
from torch_npu.testing.testcase import TestCase, run_tests

from data_cache import golden_data_cache
import mx_driving

DEFAULT_RANDOM_SEED = 20260701


@dataclass(frozen=True)
class RadiusCase:
    name: str
    purpose: str
    data_range: int
    batch_size: int
    max_points_per_batch: int
    r: int
    max_num_neighbors: int
    ndim: int


RANDOM_CASES = [
    RadiusCase(
        "large_batch_max_points_2d",
        "cover max documented batch/points shape",
        100,
        1024,
        512,
        50,
        300,
        2,
    ),
    RadiusCase("dense_radius_2d", "cover many neighbors within radius", 10, 1024, 512, 50, 300, 2),
    RadiusCase(
        "zero_radius_2d",
        "cover empty output with strict dist < r2 rule",
        100,
        100,
        256,
        0,
        30,
        2,
    ),
    RadiusCase(
        "single_neighbor_2d",
        "cover max_num_neighbors truncation to one",
        10,
        100,
        64,
        50,
        1,
        2,
    ),
    RadiusCase(
        "small_zero_radius_2d",
        "cover small input with zero radius and truncation",
        10,
        10,
        20,
        0,
        1,
        2,
    ),
    RadiusCase("many_small_batches_2d", "cover many small batches", 100, 40, 10, 50, 300, 2),
    RadiusCase("points_65_after_64_align_1d", "cover 1d points just above 64 alignment", 100, 57, 65, 20, 16, 1),
    RadiusCase("points_65_after_64_align_2d", "cover 2d points just above 64 alignment", 100, 57, 65, 20, 16, 2),
    RadiusCase("points_65_after_64_align_3d", "cover 3d points just above 64 alignment", 100, 57, 65, 20, 16, 3),
    RadiusCase("points_65_after_64_align_4d", "cover generic 4d loop just above 64 alignment", 100, 57, 65, 20, 16, 4),
    RadiusCase("four_dim_64_points", "cover generic coordinate loop with 4d points", 100, 33, 64, 20, 16, 4),
    RadiusCase("max_dim_8d", "cover max supported coordinate dimension", 100, 19, 32, 20, 16, 8),
    RadiusCase(
        "points_129_after_128_align_1d",
        "cover 1d points just above 128 alignment",
        100,
        17,
        129,
        20,
        32,
        1,
    ),
    RadiusCase(
        "points_129_after_128_align_2d",
        "cover 2d points just above 128 alignment",
        100,
        17,
        129,
        20,
        32,
        2,
    ),
    RadiusCase(
        "points_129_after_128_align_3d",
        "cover 3d points just above 128 alignment",
        100,
        17,
        129,
        20,
        32,
        3,
    ),
    RadiusCase(
        "points_129_after_128_align_4d",
        "cover generic 4d loop just above 128 alignment",
        100,
        17,
        129,
        20,
        32,
        4,
    ),
]


@golden_data_cache(__file__)
def radius_golden_python(x, y, ptr_x, ptr_y, r, max_num_neighbors):
    index_y_list = []
    index_x_list = []
    r2 = r * r
    batch_size = ptr_x.numel() - 1

    for batch_idx in range(batch_size):
        ptr_x_left = int(ptr_x[batch_idx])
        ptr_x_right = int(ptr_x[batch_idx + 1])
        ptr_y_left = int(ptr_y[batch_idx])
        ptr_y_right = int(ptr_y[batch_idx + 1])
        points_x = x[ptr_x_left:ptr_x_right]
        if points_x.numel() == 0:
            continue

        for point_y_idx in range(ptr_y_left, ptr_y_right):
            dist = torch.sum((points_x - y[point_y_idx]) ** 2, dim=1)
            local_index_x = torch.nonzero(dist < r2, as_tuple=False).flatten()[:max_num_neighbors]
            if local_index_x.numel() == 0:
                continue
            index_x = local_index_x + ptr_x_left
            index_y = torch.full_like(index_x, point_y_idx)
            index_y_list.append(index_y)
            index_x_list.append(index_x)

    if not index_x_list:
        return torch.empty((2, 0), dtype=torch.int32)
    return torch.stack([torch.cat(index_y_list), torch.cat(index_x_list)], dim=0).int()


def make_ptr(counts):
    batch_ptr = torch.tensor(counts, dtype=torch.int32).cumsum(dim=0).int()
    return torch.cat([torch.zeros([1], dtype=torch.int32), batch_ptr])


def gen_points(num_points, data_range, ndim):
    return 2 * data_range * (torch.rand([num_points, ndim]) - 0.5)


def gen_batch_ptr(batch_size, max_points_per_batch):
    batch_list = torch.randint(0, max_points_per_batch, [batch_size]).int()
    batch_ptr = torch.cumsum(batch_list, dim=0).int()
    return torch.cat([torch.zeros([1]).int(), batch_ptr])


@golden_data_cache(__file__)
def gen_inputs(data_range, batch_size, max_points_per_batch, ndim, seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    ptr_x = gen_batch_ptr(batch_size, max_points_per_batch)
    ptr_y = gen_batch_ptr(batch_size, max_points_per_batch)
    x = gen_points(int(ptr_x[-1]), data_range, ndim)
    y = gen_points(int(ptr_y[-1]), data_range, ndim)
    return x, y, ptr_x, ptr_y


class TestRadius(TestCase):
    def _assert_radius(self, x, y, ptr_x, ptr_y, r, max_num_neighbors):
        out_cpu = radius_golden_python(x, y, ptr_x, ptr_y, r, max_num_neighbors)
        out_npu = mx_driving.radius(x.npu(), y.npu(), ptr_x.npu(), ptr_y.npu(), r, max_num_neighbors).cpu()
        self.assertRtolEqual(out_cpu, out_npu)

    def test_radius_truncates_neighbors_and_skips_empty_batch(self):
        # The middle batch is empty; max_num_neighbors=1 checks per-y truncation.
        x = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [10.0, 10.0],
                [11.0, 10.0],
            ]
        )
        y = torch.tensor(
            [
                [0.5, 0.0],
                [3.0, 0.0],
                [10.2, 10.0],
            ]
        )
        ptr_x = make_ptr([3, 0, 2])
        ptr_y = make_ptr([2, 0, 1])
        self._assert_radius(x, y, ptr_x, ptr_y, 1.1, 1)

    def test_radius_zero_radius_returns_empty_result(self):
        # The operator uses strict distance comparison, so r=0 should select nothing.
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ptr_x = make_ptr([3])
        ptr_y = make_ptr([2])
        self._assert_radius(x, y, ptr_x, ptr_y, 0.0, 3)

    def test_radius_rejects_negative_max_num_neighbors(self):
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[0.0, 0.0]])
        ptr_x = make_ptr([1])
        ptr_y = make_ptr([1])

        with self.assertRaisesRegex(RuntimeError, "max_num_neighbors must be non-negative"):
            mx_driving.radius(x, y, ptr_x, ptr_y, 1.0, -1)

    def test_radius_random(self):
        for case in RANDOM_CASES:
            with self.subTest(case=case.name, purpose=case.purpose):
                x, y, ptr_x, ptr_y = gen_inputs(
                    case.data_range,
                    case.batch_size,
                    case.max_points_per_batch,
                    case.ndim,
                    DEFAULT_RANDOM_SEED,
                )
                self._assert_radius(x, y, ptr_x, ptr_y, case.r, case.max_num_neighbors)

    def test_radius_batch_size_sweep(self):
        for batch_size in range(1, 1025, 97):
            with self.subTest(batch_size=batch_size):
                x, y, ptr_x, ptr_y = gen_inputs(100, batch_size, 50, 2, DEFAULT_RANDOM_SEED)
                self._assert_radius(x, y, ptr_x, ptr_y, 50, 300)

    def test_radius_points_per_batch_sweep(self):
        for max_points_per_batch in range(2, 513, 47):
            with self.subTest(max_points_per_batch=max_points_per_batch):
                x, y, ptr_x, ptr_y = gen_inputs(100, 50, max_points_per_batch, 2, DEFAULT_RANDOM_SEED)
                self._assert_radius(x, y, ptr_x, ptr_y, 50, 300)


if __name__ == "__main__":
    run_tests()
