"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

DrivingSDK NPU operator performance collection and comparison utility.

ops_performance 工具使用指南请见 docs/zh/tools/ops_perf.md
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mx_driving.tools.ops_perf import OpsPerfEnvironmentError, OpsPerfUsageError, ops_performance, set_perf
from mx_driving.tools.ops_perf import ops_perf


class TestOpsPerformance(unittest.TestCase):
    def setUp(self):
        ops_perf._CsvWriter._paths.clear()
        ops_perf._PAIR_RESULTS.clear()
        ops_perf._CASE_OCCURRENCES.clear()
        set_perf()

    @staticmethod
    def config(output_dir, **kwargs):
        result = {
            "warmup": 0,
            "repeat": 1,
            "output_dir": output_dir,
            "synchronize_fn": lambda: None,
            "quiet": True,
            "timer": "host",
            "details": True,
        }
        result.update(kwargs)
        return result

    @staticmethod
    def read_rows(output_dir, mode):
        reports = list(Path(output_dir).glob("*_{0}_*.csv".format(mode)))
        if len(reports) != 1:
            raise AssertionError("expected one {} report, got {}".format(mode, reports))
        with reports[0].open(encoding="utf-8-sig", newline="") as stream:
            return reports[0], list(csv.DictReader(stream))

    def test_baseline_mode_records_times_and_meta(self):
        calls = []
        with tempfile.TemporaryDirectory() as output_dir:
            result = ops_performance(
                lambda value: calls.append(value) or value * 2,
                4,
                _perf=self.config(output_dir, warmup=1, repeat=1, case_id="baseline_case", metadata={"shape": [4]}),
            )
            _, rows = self.read_rows(output_dir, "baseline")
            row = rows[0]
            meta = json.loads(row["Meta"])

        self.assertEqual(result, 8)
        self.assertEqual(calls, [4, 4])
        self.assertEqual(row["Case ID"], "baseline_case")
        self.assertNotIn("Case Name", row)
        self.assertEqual(row["Device Time (us)"], "")
        self.assertTrue(row["Host + Device Time (us)"])
        self.assertNotIn("Status", row)
        self.assertEqual(meta["shape"], [4])
        self.assertIn("git_commit", meta)
        self.assertIn("pytorch_version", meta)

    def test_default_sampling_is_five_warmups_and_five_samples(self):
        calls = []
        with tempfile.TemporaryDirectory() as output_dir:
            ops_performance(
                lambda: calls.append(1),
                _perf={
                    "output_dir": output_dir,
                    "synchronize_fn": lambda: None,
                    "quiet": True,
                    "timer": "host",
                    "details": True,
                },
            )
            _, rows = self.read_rows(output_dir, "baseline")
            meta = json.loads(rows[0]["Meta"])
        self.assertEqual(len(calls), 10)
        self.assertEqual(meta["warmup"], 5)
        self.assertEqual(meta["repeat"], 5)

    def test_multiple_cases_in_one_script_share_one_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            for case_id in ("bev_pool_v3_with_depth", "bev_pool_v3_without_depth", "bev_pool_v3_grad"):
                ops_performance(lambda: None, _perf=self.config(output_dir, case_id=case_id))
            report, rows = self.read_rows(output_dir, "baseline")

        self.assertTrue(report.name.startswith("test_ops_perf_baseline_"))
        self.assertEqual(
            [row["Case ID"] for row in rows],
            ["bev_pool_v3_with_depth", "bev_pool_v3_without_depth", "bev_pool_v3_grad"],
        )

    def test_historical_performance_comparison_supports_multiple_files(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline_one = Path(output_dir) / "baseline_one.csv"
            baseline_two = Path(output_dir) / "baseline_two.csv"
            fields = ops_perf._SAVE_FIELDS
            for path, value in ((baseline_one, "20.000"), (baseline_two, "30.000")):
                with path.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({"Case ID": "compare_case", "Device Time (us)": value})

            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 10.0, 2)):
                ops_performance(
                    lambda: None,
                    _perf=self.config(
                        output_dir,
                        timer="device",
                        mode="performance",
                        case_id="compare_case",
                        baseline_files=[baseline_one, baseline_two],
                    ),
                )
            _, rows = self.read_rows(output_dir, "performance")

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["Baseline Device Time (us)"] for row in rows], ["20.000", "30.000"])
        self.assertEqual([row["Candidate Device Time (us)"] for row in rows], ["10.000", "10.000"])
        self.assertEqual([row["Device Performance Comparison"] for row in rows], ["+100.00%", "+200.00%"])
        self.assertTrue(all(row["Device Verdict"] == "PASS" for row in rows))

    def test_baseline_file_only_compares_matching_case(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "five_cases.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
                for index in range(5):
                    writer.writerow({"Case ID": "case_{}".format(index), "Device Time (us)": "20.000"})
            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 10.0, 1)):
                ops_performance(
                    lambda: None,
                    _perf=self.config(
                        output_dir,
                        timer="device",
                        mode="performance",
                        case_id="case_3",
                        baseline_files=baseline,
                    ),
                )
            _, rows = self.read_rows(output_dir, "performance")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Case ID"], "case_3")

    def test_repeated_case_ids_match_baseline_by_occurrence(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "repeated_cases.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
                for value in (3.396, 15.384, 18.900, 50.764, 81.264):
                    writer.writerow({"Case ID": "bev_pool_v3", "Device Time (us)": value})

            candidates = iter((3.280, 18.084, 18.492, 50.940, 81.188))

            def profile_samples(*_args):
                return None, next(candidates), 1

            with mock.patch.object(ops_perf, "_profile_device_samples", side_effect=profile_samples):
                for _ in range(5):
                    ops_performance(
                        lambda: None,
                        _perf=self.config(
                            output_dir,
                            timer="device",
                            mode="performance",
                            case_id="bev_pool_v3",
                            baseline_files=baseline,
                        ),
                    )
            _, rows = self.read_rows(output_dir, "performance")

        self.assertEqual(
            [row["Baseline Device Time (us)"] for row in rows],
            ["3.396", "15.384", "18.900", "50.764", "81.264"],
        )

    def test_regression_threshold(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "baseline.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
                writer.writerow({"Case ID": "regression", "Device Time (us)": "10.000"})
            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 12.0, 1)):
                ops_performance(
                    lambda: None,
                    _perf=self.config(
                        output_dir,
                        timer="device",
                        mode="performance",
                        case_id="regression",
                        baseline_files=baseline,
                        threshold_pct=5,
                    ),
                )
            _, rows = self.read_rows(output_dir, "performance")
        self.assertEqual(rows[0]["Device Performance Comparison"], "-16.67%")
        self.assertEqual(rows[0]["Device Regression (%)"], "20.00%")
        self.assertEqual(rows[0]["Device Verdict"], "REGRESSION")

    def test_in_process_baseline_candidate_pair(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 20.0, 1)):
                ops_performance(
                    lambda: None,
                    _perf=self.config(
                        output_dir, timer="device", mode="performance", flag="baseline", case_id="pair_case"
                    ),
                )
            self.assertEqual(list(Path(output_dir).glob("*_performance_*.csv")), [])
            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 10.0, 1)):
                ops_performance(
                    lambda: None,
                    _perf=self.config(
                        output_dir, timer="device", mode="performance", flag="candidate", case_id="pair_case"
                    ),
                )
            _, rows = self.read_rows(output_dir, "performance")
        self.assertEqual(rows[0]["Baseline File"], "in-process:baseline")
        self.assertEqual(rows[0]["Device Performance Comparison"], "+100.00%")

    def test_missing_baseline_is_recorded(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "empty.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
            ops_performance(
                lambda: None,
                _perf=self.config(output_dir, mode="performance", case_id="missing", baseline_files=baseline),
            )
            _, rows = self.read_rows(output_dir, "performance")
        self.assertEqual(rows[0]["Host Verdict"], "MISSING_BASELINE")

    def test_exception_is_recorded_then_reraised(self):
        def failed_op():
            raise RuntimeError("kernel failed")

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                ops_performance(failed_op, _perf=self.config(output_dir, case_id="failed_case"))
            _, rows = self.read_rows(output_dir, "baseline")
            meta = json.loads(rows[0]["Meta"])
        self.assertNotIn("Status", rows[0])
        self.assertEqual(meta["error_type"], "RuntimeError")
        self.assertIn("kernel failed", meta["traceback"])

    def test_oom_is_recorded(self):
        def oom_op():
            raise RuntimeError("NPU out of memory")

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(RuntimeError):
                ops_performance(oom_op, _perf=self.config(output_dir, case_id="oom_case"))
            _, rows = self.read_rows(output_dir, "baseline")
        self.assertNotIn("Status", rows[0])

    def test_default_output_is_beside_script(self):
        script_path = Path(__file__).resolve()
        expected = script_path.parent / "performance_data" / script_path.stem
        with mock.patch.object(ops_perf, "_find_caller", return_value=(script_path, "test_case")):
            root = ops_perf._output_root(script_path, None)
        self.assertEqual(root, expected)

    def test_configuration_errors(self):
        invalid = [
            ({"mode": "bad"}, "mode must be"),
            ({"flag": "bad", "mode": "performance"}, "performance flag"),
            ({"warmup": -1}, "warmup must be"),
            ({"repeat": 0}, "repeat must be"),
            ({"metadata": []}, "metadata must be"),
            ({"threshold_pct": "bad"}, "threshold_pct must be"),
            ({"timer": "bad"}, "timer must be"),
            ({"details": "yes"}, "details must be"),
            ({"case_id": ""}, "case_id must be"),
            ({"unknown": 1}, "unsupported _perf option"),
        ]
        for extra, message in invalid:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as output_dir:
                config = self.config(output_dir, **extra)
                with self.assertRaisesRegex(OpsPerfUsageError, message):
                    ops_performance(lambda: None, _perf=config)

    def test_missing_npu_environment(self):
        with mock.patch.object(ops_perf.torch, "npu", None, create=True):
            with self.assertRaises(OpsPerfEnvironmentError):
                ops_perf._default_synchronize()

    def test_kernel_details_are_summed(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run" / "ASCEND_PROFILER_OUTPUT"
            output.mkdir(parents=True)
            with (output / "kernel_details.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Name", "Duration(us)"])
                writer.writeheader()
                writer.writerows(
                    [{"Name": "kernel_a", "Duration(us)": "2.5"}, {"Name": "kernel_b", "Duration(us)": "3.0"}]
                )
            duration, count = ops_perf._read_kernel_details(root)
        self.assertEqual(duration, 5.5)
        self.assertEqual(count, 2)

    def test_device_timer_only_profiles_samples(self):
        calls = []

        def profile_samples(function, args, kwargs, synchronize, repeat):
            synchronize()
            result = None
            for _ in range(repeat):
                result = function(*args, **kwargs)
            return result, 7.0, 2

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(ops_perf, "_profile_device_samples", side_effect=profile_samples) as profiler:
                ops_performance(
                    lambda: calls.append(1),
                    _perf=self.config(output_dir, timer="device", warmup=2, repeat=3),
                )
                _, rows = self.read_rows(output_dir, "baseline")
                meta = json.loads(rows[0]["Meta"])
        self.assertEqual(len(calls), 5)
        self.assertEqual(profiler.call_count, 1)
        self.assertEqual(rows[0]["Device Time (us)"], "7.000")
        self.assertEqual(rows[0]["Host + Device Time (us)"], "")
        self.assertEqual(meta["kernel_counts"], [2])

    def test_both_timer_runs_separate_device_and_host_samples(self):
        calls = []

        def profile_samples(function, args, kwargs, synchronize, repeat):
            result = None
            for _ in range(repeat):
                calls.append("device")
                result = function(*args, **kwargs)
            return result, 7.0, 2

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(ops_perf, "_profile_device_samples", side_effect=profile_samples):
                with mock.patch.object(ops_perf.time, "perf_counter_ns", side_effect=[0, 10_000, 20_000, 30_000]):
                    ops_performance(
                        lambda: calls.append("call"),
                        _perf=self.config(output_dir, timer="both", warmup=1, repeat=2),
                    )
                    _, rows = self.read_rows(output_dir, "baseline")

        self.assertEqual(calls.count("call"), 5)
        self.assertEqual(calls.count("device"), 2)
        self.assertEqual(rows[0]["Device Time (us)"], "7.000")
        self.assertEqual(rows[0]["Host + Device Time (us)"], "10.000")

    def test_both_timer_writes_one_combined_performance_row(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "both_baseline.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {"Case ID": "both_case", "Device Time (us)": "20.000", "Host + Device Time (us)": "40.000"}
                )
            with mock.patch.object(ops_perf, "_profile_device_samples", return_value=(None, 10.0, 1)):
                with mock.patch.object(ops_perf.time, "perf_counter_ns", side_effect=[0, 20_000]):
                    ops_performance(
                        lambda: None,
                        _perf=self.config(
                            output_dir,
                            timer="both",
                            mode="performance",
                            case_id="both_case",
                            baseline_files=baseline,
                        ),
                    )
            _, rows = self.read_rows(output_dir, "performance")

        self.assertEqual(len(rows), 1)
        self.assertNotIn("Timer", rows[0])
        self.assertEqual(rows[0]["Device Performance Comparison"], "+100.00%")
        self.assertEqual(rows[0]["Host Performance Comparison"], "+100.00%")

    def test_host_performance_uses_host_baseline(self):
        with tempfile.TemporaryDirectory() as output_dir:
            baseline = Path(output_dir) / "host_baseline.csv"
            with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ops_perf._SAVE_FIELDS)
                writer.writeheader()
                writer.writerow({"Case ID": "host_case", "Host + Device Time (us)": "20.000"})
            with mock.patch.object(ops_perf.time, "perf_counter_ns", side_effect=[0, 10_000]):
                ops_performance(
                    lambda: None,
                    _perf=self.config(output_dir, mode="performance", case_id="host_case", baseline_files=baseline),
                )
            _, rows = self.read_rows(output_dir, "performance")
        self.assertNotIn("Timer", rows[0])
        self.assertEqual(rows[0]["Baseline Host Time (us)"], "20.000")
        self.assertEqual(rows[0]["Candidate Host Time (us)"], "10.000")
        self.assertNotIn("Baseline Device Time (us)", rows[0])
        self.assertEqual(rows[0]["Host Performance Comparison"], "+100.00%")

    def test_default_csv_is_compact(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ops_performance(
                lambda: None,
                _perf={
                    "timer": "host",
                    "warmup": 0,
                    "repeat": 1,
                    "output_dir": output_dir,
                    "synchronize_fn": lambda: None,
                    "quiet": True,
                },
            )
            _, rows = self.read_rows(output_dir, "baseline")
        self.assertEqual(list(rows[0]), ops_perf._SAVE_FIELDS)
        self.assertNotIn("Case Name", rows[0])
        self.assertNotIn("Meta", rows[0])

    def test_global_config_and_call_override(self):
        with tempfile.TemporaryDirectory() as output_dir:
            set_perf(timer="host", warmup=0, repeat=1, output_dir=output_dir, synchronize_fn=lambda: None, quiet=True)
            ops_performance(lambda: None, _perf={"case_id": "global_case", "details": True})
            _, rows = self.read_rows(output_dir, "baseline")
        self.assertEqual(rows[0]["Case ID"], "global_case")
        self.assertIn("Meta", rows[0])


if __name__ == "__main__":
    unittest.main()
