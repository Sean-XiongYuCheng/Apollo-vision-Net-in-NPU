"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

precision_compare 工具使用指南请见 docs/zh/tools/precision_compare.md
"""

import unittest

import numpy as np

from mx_driving.tools.precision_compare import assert_precision, check_precision, precision_compare


class TestPrecisionCompare(unittest.TestCase):
    def test_float_result_contains_error_details(self):
        prediction = np.array([1.0, 2.0, 3.1], dtype=np.float32)
        benchmark = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = precision_compare(
            prediction,
            benchmark,
            err_threshold=1e-3,
            err_ratio=0.34,
            max_diff_hard=1.0,
            verbose=False,
        )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.pass_rate, 200.0 / 3.0)
        np.testing.assert_array_equal(result.error_indices, [2])
        np.testing.assert_allclose(result.error_benchmarks, [3.0])
        np.testing.assert_allclose(result.error_predictions, [3.1], rtol=1e-6)
        self.assertEqual(result.total_count, 3)

    def test_integer_inputs_require_exact_match(self):
        result = precision_compare([1, 2, 4], [1, 2, 3], verbose=False)

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.pass_rate, 200.0 / 3.0)
        np.testing.assert_array_equal(result.error_indices, [2])
        np.testing.assert_array_equal(result.error_abs_diffs, [1.0])

    def test_matching_nan_and_inf_are_accepted_and_counted(self):
        prediction = np.array([np.nan, np.inf, -np.inf, 1.0])
        benchmark = np.array([np.nan, np.inf, -np.inf, 1.0])

        result = precision_compare(prediction, benchmark, verbose=False)

        self.assertTrue(result.passed)
        self.assertEqual(result.nan_count_pred, 1)
        self.assertEqual(result.nan_count_bench, 1)
        self.assertEqual(result.inf_count_pred, 2)
        self.assertEqual(result.inf_count_bench, 2)
        self.assertEqual(result.error_indices.size, 0)

    def test_convenience_functions(self):
        self.assertTrue(assert_precision([1.0, 2.0], [1.0, 2.0], verbose=False))
        self.assertIsNone(check_precision([1.0, 2.0], [1.0, 2.0], verbose=False))
        with self.assertRaisesRegex(AssertionError, "Precision check failed"):
            check_precision([1.0, 2.0], [1.0, 3.0], verbose=False)

    def test_invalid_inputs(self):
        invalid_cases = [
            ((np.array([1.0]), np.array([1.0, 2.0])), "Shape mismatch"),
            ((np.array([]), np.array([])), "Input data is empty"),
        ]
        for args, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                precision_compare(*args, verbose=False)

        with self.assertRaisesRegex(ValueError, "err_threshold must be"):
            precision_compare([1.0], [1.0], err_threshold=-1.0, verbose=False)
        with self.assertRaisesRegex(ValueError, "err_ratio must be"):
            precision_compare([1.0], [1.0], err_ratio=1.1, verbose=False)
        with self.assertRaisesRegex(TypeError, "Unsupported data type"):
            precision_compare({"value": 1.0}, [1.0], verbose=False)


if __name__ == "__main__":
    unittest.main()
