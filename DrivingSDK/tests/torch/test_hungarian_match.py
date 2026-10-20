import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


class TestHungarian(TestCase):
    def test_untranspose_hungarian_match(self, device="npu"):
        for n in [1, 5, 10, 20, 100, 400, 900]:
            for m in [1, 3, 10, 70, 100, 300, 900]:
                if n > m:
                    n, m = m, n
                cost_matrix = np.random.uniform(-100, 100, (n, m)).astype(np.float32)
                expected_row, expected_col = linear_sum_assignment(cost_matrix)
                row_ind, col_ind = mx_driving.hungarian_match(torch.from_numpy(cost_matrix).npu())
                self.assertRtolEqual(row_ind.cpu().numpy(), expected_row.astype(np.int32))
                self.assertRtolEqual(col_ind.cpu().numpy(), expected_col.astype(np.int32))

    def test_transpose_hungarian_match(self, device="npu"):
        for n in [1, 5, 10, 20, 100, 400, 900]:
            for m in [2, 6, 40, 70, 200, 700]:
                if n < m:
                    n, m = m, n
                cost_matrix = np.random.uniform(-100, 100, (n, m)).astype(np.float32)
                expected_row, expected_col = linear_sum_assignment(cost_matrix)
                row_ind, col_ind = mx_driving.hungarian_match(torch.from_numpy(cost_matrix).npu())
                self.assertRtolEqual(row_ind.cpu().numpy(), expected_row.astype(np.int32))
                self.assertRtolEqual(col_ind.cpu().numpy(), expected_col.astype(np.int32))


if __name__ == "__main__":
    run_tests()
