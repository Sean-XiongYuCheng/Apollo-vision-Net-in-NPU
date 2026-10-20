"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

DrivingSDK NPU operator performance collection and comparison utility.

ops_performance 工具使用指南请见 docs/zh/tools/ops_perf.md
"""

import atexit
import contextlib
import csv
import datetime as dt
import inspect
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
import traceback
import warnings
from pathlib import Path

import torch


_SAVE_FIELDS = ["Case ID", "Device Time (us)", "Host + Device Time (us)"]
_SAVE_DETAIL_FIELDS = _SAVE_FIELDS + ["Meta"]
_DEVICE_PERFORMANCE_FIELDS = [
    "Baseline Device Time (us)",
    "Candidate Device Time (us)",
    "Device Performance Comparison",
]
_HOST_PERFORMANCE_FIELDS = [
    "Baseline Host Time (us)",
    "Candidate Host Time (us)",
    "Host Performance Comparison",
]
_PAIR_RESULTS = {}
_PAIR_LOCK = threading.RLock()
_GLOBAL_CONFIG = {}
_CONFIG_LOCK = threading.RLock()
_CASE_OCCURRENCES = {}
_CASE_OCCURRENCE_LOCK = threading.RLock()


class OpsPerfError(RuntimeError):
    """Base OpsPerf exception."""


class OpsPerfUsageError(OpsPerfError):
    """Raised for invalid benchmark arguments."""


class OpsPerfEnvironmentError(OpsPerfError):
    """Raised when NPU timing is unavailable."""


def _performance_fields(timer, details=False):
    fields = ["Case ID"]
    if timer in ("device", "both"):
        fields.extend(_DEVICE_PERFORMANCE_FIELDS)
    if timer in ("host", "both"):
        fields.extend(_HOST_PERFORMANCE_FIELDS)
    if details:
        if timer in ("device", "both"):
            fields.extend(["Device Regression (%)", "Device Verdict"])
        if timer in ("host", "both"):
            fields.extend(["Host Regression (%)", "Host Verdict"])
        fields.extend(["Baseline File", "Meta"])
    return fields


def _safe_filename(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    return value or "ops_perf"


def _find_caller():
    current = Path(__file__).resolve()
    for frame in inspect.stack()[2:]:
        path = Path(frame.filename).resolve()
        if path != current:
            return path, frame.function
    return Path(sys.argv[0] or "ops_perf.py").resolve(), "<module>"


def _default_synchronize():
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise OpsPerfEnvironmentError("torch.npu is unavailable")
    npu.synchronize()


def _device_name():
    try:
        return str(torch.npu.get_device_name(torch.npu.current_device()))
    except Exception:
        return "unknown"


def _git_commit(script_path):
    try:
        import subprocess

        git = shutil.which("git")
        if git is None:
            return "unknown"
        process = subprocess.run(
            [git, "-C", str(script_path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return process.stdout.strip() if process.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _output_root(script_path, output_dir):
    root = Path(output_dir) if output_dir else script_path.parent / "performance_data" / script_path.stem
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


class _CsvWriter:
    _lock = threading.RLock()
    _paths = {}

    @classmethod
    def path_for(cls, script_path, output_dir, mode, details=False, timer=None):
        root = _output_root(script_path, output_dir)
        rank = os.environ.get("RANK", "0")
        schema = timer if mode == "performance" else None
        key = (str(root), str(script_path), mode, rank, details, schema)
        with cls._lock:
            if key not in cls._paths:
                timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                rank_suffix = "" if os.environ.get("WORLD_SIZE", "1") == "1" else "_rank{}".format(rank)
                detail_suffix = "_details" if details else ""
                timer_suffix = "_{}".format(timer) if mode == "performance" and timer else ""
                cls._paths[key] = root / "{}_{}_{}{}{}{}.csv".format(
                    _safe_filename(script_path.stem), mode, timestamp, rank_suffix, timer_suffix, detail_suffix
                )
            return cls._paths[key]

    @classmethod
    def append(cls, script_path, output_dir, mode, row, details=False, timer=None):
        path = cls.path_for(script_path, output_dir, mode, details, timer)
        fields = (
            _SAVE_DETAIL_FIELDS
            if mode == "baseline" and details
            else _SAVE_FIELDS
            if mode == "baseline"
            else _performance_fields(timer, details)
        )
        with cls._lock:
            is_new = not path.exists() or path.stat().st_size == 0
            with path.open("a", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                writer.writerow({field: row.get(field, "") for field in fields})
        return path


def _normalize_baseline_files(value):
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def set_perf(_perf=None, **kwargs):
    """Set process-wide defaults; per-call _perf values take precedence."""
    if _perf is not None and not isinstance(_perf, dict):
        raise OpsPerfUsageError("set_perf expects a dict or keyword options")
    config = dict(_perf or {})
    config.update(kwargs)
    _validate_config(config)
    with _CONFIG_LOCK:
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(config)


def _validate_config(config):
    supported = {
        "case_id",
        "warmup",
        "repeat",
        "mode",
        "flag",
        "baseline_files",
        "threshold_pct",
        "output_dir",
        "metadata",
        "quiet",
        "synchronize_fn",
        "timer",
        "details",
    }
    unknown = sorted(set(config) - supported)
    if unknown:
        raise OpsPerfUsageError("unsupported _perf option(s): {}".format(", ".join(unknown)))
    warmup = config.get("warmup", 5)
    repeat = config.get("repeat", 5)
    mode = config.get("mode", "baseline")
    flag = config.get("flag")
    timer = config.get("timer", "device")
    details = config.get("details", False)
    if not isinstance(warmup, int) or warmup < 0:
        raise OpsPerfUsageError("warmup must be a non-negative integer")
    if not isinstance(repeat, int) or repeat <= 0:
        raise OpsPerfUsageError("repeat must be a positive integer")
    if mode not in ("baseline", "performance"):
        raise OpsPerfUsageError("mode must be 'baseline' or 'performance'")
    if mode == "performance" and flag not in (None, "baseline", "candidate"):
        raise OpsPerfUsageError("performance flag must be 'baseline' or 'candidate'")
    if timer not in ("device", "host", "both"):
        raise OpsPerfUsageError("timer must be 'device', 'host', or 'both'")
    if not isinstance(details, bool):
        raise OpsPerfUsageError("details must be a bool")
    if config.get("metadata") is not None and not isinstance(config["metadata"], dict):
        raise OpsPerfUsageError("metadata must be a dict or None")
    try:
        threshold_pct = float(config.get("threshold_pct", 5.0))
    except (TypeError, ValueError) as error:
        raise OpsPerfUsageError("threshold_pct must be numeric") from error
    return warmup, repeat, mode, flag, threshold_pct, timer


def _read_baselines(paths, case_id, timer, occurrence):
    matches = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise OpsPerfUsageError("baseline CSV does not exist: {}".format(path))
        matched_index = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Case ID") != case_id:
                    continue
                if matched_index != occurrence:
                    matched_index += 1
                    continue
                field = "Device Time (us)" if timer == "device" else "Host + Device Time (us)"
                value = row.get(field)
                try:
                    matches.append((str(path), float(value)))
                except (TypeError, ValueError):
                    pass
                break
    return matches


def _read_both_baselines(paths, case_id, occurrence):
    matches = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise OpsPerfUsageError("baseline CSV does not exist: {}".format(path))
        matched_index = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Case ID") != case_id:
                    continue
                if matched_index != occurrence:
                    matched_index += 1
                    continue
                try:
                    device_us = float(row.get("Device Time (us)", ""))
                    host_us = float(row.get("Host + Device Time (us)", ""))
                except (TypeError, ValueError):
                    break
                matches.append((str(path), device_us, host_us))
                break
    return matches


def _next_case_occurrence(script_path, case_id, timer, baseline_files):
    files = tuple(str(Path(path).expanduser().resolve()) for path in baseline_files)
    key = (str(script_path), case_id, timer, files, os.environ.get("RANK", "0"))
    with _CASE_OCCURRENCE_LOCK:
        occurrence = _CASE_OCCURRENCES.get(key, 0)
        _CASE_OCCURRENCES[key] = occurrence + 1
    return occurrence


def _comparison_row(result, baseline_us, baseline_file, threshold_pct):
    timer = result["timer"]
    candidate_us = result["device_us"] if timer == "device" else result["host_device_us"]
    change_pct = (baseline_us / candidate_us - 1.0) * 100.0 if candidate_us else float("inf")
    regression_pct = max(0.0, (candidate_us / baseline_us - 1.0) * 100.0) if baseline_us else float("inf")
    title = "Device" if timer == "device" else "Host"
    return {
        "Case ID": result["case_id"],
        "Baseline {} Time (us)".format(title): "{:.3f}".format(baseline_us),
        "Candidate {} Time (us)".format(title): "{:.3f}".format(candidate_us),
        "{} Performance Comparison".format(title): "{:+.2f}%".format(change_pct),
        "{} Regression (%)".format(title): "{:.2f}%".format(regression_pct),
        "{} Verdict".format(title): "REGRESSION" if regression_pct > threshold_pct else "PASS",
        "Baseline File": baseline_file,
        "Meta": result["meta"],
    }


def _both_comparison_row(result, baseline_device_us, baseline_host_us, baseline_file, threshold_pct):
    device_result = dict(result, timer="device")
    host_result = dict(result, timer="host")
    row = _comparison_row(device_result, baseline_device_us, baseline_file, threshold_pct)
    row.update(_comparison_row(host_result, baseline_host_us, baseline_file, threshold_pct))
    return row


def _write_result(result, config, mode, flag, threshold_pct):
    script_path = result["script_path"]
    output_dir = config.get("output_dir")
    if mode == "baseline":
        return _CsvWriter.append(script_path, output_dir, mode, result["baseline_row"], config.get("details", False))

    baseline_files = _normalize_baseline_files(config.get("baseline_files"))
    if baseline_files:
        occurrence = _next_case_occurrence(script_path, result["case_id"], result["timer"], baseline_files)
        if result["timer"] == "both":
            matches = _read_both_baselines(baseline_files, result["case_id"], occurrence)
        else:
            matches = _read_baselines(baseline_files, result["case_id"], result["timer"], occurrence)
        if not matches:
            if result["timer"] == "both":
                row = _both_comparison_row(result, result["device_us"], result["host_device_us"], "", threshold_pct)
            else:
                candidate_us = result["device_us"] if result["timer"] == "device" else result["host_device_us"]
                row = _comparison_row(result, candidate_us, "", threshold_pct)
            for field in list(row):
                if field.startswith("Baseline ") or field.endswith("Performance Comparison"):
                    row[field] = ""
                if field.endswith("Verdict"):
                    row[field] = "MISSING_BASELINE"
            return _CsvWriter.append(script_path, output_dir, mode, row, config.get("details", False), result["timer"])
        path = None
        for match in matches:
            if result["timer"] == "both":
                baseline_file, baseline_device_us, baseline_host_us = match
                row = _both_comparison_row(result, baseline_device_us, baseline_host_us, baseline_file, threshold_pct)
            else:
                baseline_file, baseline_us = match
                row = _comparison_row(result, baseline_us, baseline_file, threshold_pct)
            path = _CsvWriter.append(
                script_path,
                output_dir,
                mode,
                row,
                config.get("details", False),
                result["timer"],
            )
        return path

    pair_flag = flag or "candidate"
    key = (str(script_path), result["case_id"], result["timer"], os.environ.get("RANK", "0"))
    with _PAIR_LOCK:
        pair = _PAIR_RESULTS.setdefault(key, {})
        if pair_flag in pair:
            raise OpsPerfUsageError("duplicate performance flag for case '{}'".format(result["case_id"]))
        pair[pair_flag] = (result, config, threshold_pct)
        if "baseline" not in pair or "candidate" not in pair:
            return None
        baseline_result, _, _ = pair["baseline"]
        candidate_result, candidate_config, candidate_threshold = pair["candidate"]
        del _PAIR_RESULTS[key]
    if candidate_result["timer"] == "both":
        row = _both_comparison_row(
            candidate_result,
            baseline_result["device_us"],
            baseline_result["host_device_us"],
            "in-process:baseline",
            candidate_threshold,
        )
    else:
        baseline_us = (
            baseline_result["device_us"] if baseline_result["timer"] == "device" else baseline_result["host_device_us"]
        )
        row = _comparison_row(candidate_result, baseline_us, "in-process:baseline", candidate_threshold)
    return _CsvWriter.append(
        script_path,
        candidate_config.get("output_dir"),
        mode,
        row,
        candidate_config.get("details", False),
        candidate_result["timer"],
    )


def _read_kernel_details(root):
    paths = list(Path(root).glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(paths) != 1:
        raise OpsPerfEnvironmentError("expected one profiler kernel_details.csv, found {}".format(len(paths)))
    durations = []
    with paths[0].open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                duration = float(row.get("Duration(us)", ""))
            except (TypeError, ValueError) as error:
                raise OpsPerfEnvironmentError("invalid Duration(us) in profiler output") from error
            if not math.isfinite(duration) or duration < 0:
                raise OpsPerfEnvironmentError("invalid Duration(us) in profiler output")
            durations.append(duration)
    if not durations:
        raise OpsPerfEnvironmentError("profiler output contains no NPU kernels")
    return sum(durations), len(durations)


def _profile_device_samples(function, args, kwargs, synchronize, repeat):
    try:
        import torch_npu
    except ImportError as error:
        raise OpsPerfEnvironmentError("torch_npu.profiler is unavailable") from error

    profiler = getattr(torch_npu, "profiler", None)
    if profiler is None:
        raise OpsPerfEnvironmentError("torch_npu.profiler is unavailable")
    root = Path(tempfile.mkdtemp(prefix="ops_perf_"))
    result_value = None
    previous_log_level = logging.root.manager.disable
    try:
        synchronize()
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.disable(logging.WARNING)
                with profiler.profile(
                    activities=[profiler.ProfilerActivity.NPU],
                    experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0),
                    on_trace_ready=profiler.tensorboard_trace_handler(str(root)),
                ):
                    for _ in range(repeat):
                        result_value = function(*args, **kwargs)
                    synchronize()
        total_us, kernel_count = _read_kernel_details(root)
        return result_value, total_us / repeat, kernel_count
    finally:
        logging.disable(previous_log_level)
        shutil.rmtree(root, ignore_errors=True)


def ops_performance(function, *args, **kwargs):
    """Benchmark one callable, persist the result, and optionally compare it."""
    if not callable(function):
        raise OpsPerfUsageError("ops_performance expects a callable as its first argument")
    call_config = kwargs.pop("_perf", None) or {}
    if not isinstance(call_config, dict):
        raise OpsPerfUsageError("_perf must be a dict or None")
    with _CONFIG_LOCK:
        config = dict(_GLOBAL_CONFIG)
    config.update(call_config)
    warmup, repeat, mode, flag, threshold_pct, timer = _validate_config(config)

    script_path, case_name = _find_caller()
    default_case_id = getattr(function, "__name__", type(function).__name__)
    case_id = config["case_id"] if "case_id" in config else default_case_id
    if not isinstance(case_id, str) or not case_id:
        raise OpsPerfUsageError("case_id must be a non-empty string")

    synchronize = config.get("synchronize_fn") or _default_synchronize
    device_samples = []
    host_device_samples = []
    kernel_counts = []
    result_value = None
    failure = None
    failure_traceback = ""

    try:
        for _ in range(warmup):
            synchronize()
            result_value = function(*args, **kwargs)
            synchronize()
        if timer in ("device", "both"):
            result_value, device_us, kernel_count = _profile_device_samples(function, args, kwargs, synchronize, repeat)
            device_samples.append(device_us)
            kernel_counts.append(kernel_count)
        if timer in ("host", "both"):
            for _ in range(repeat):
                synchronize()
                host_start_ns = time.perf_counter_ns()
                result_value = function(*args, **kwargs)
                synchronize()
                host_device_samples.append((time.perf_counter_ns() - host_start_ns) / 1_000.0)
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()

    status = "SUCCESS" if failure is None else "OOM" if "out of memory" in str(failure).lower() else "ERROR"
    device_us = statistics.mean(device_samples) if device_samples else 0.0
    host_device_us = statistics.mean(host_device_samples) if host_device_samples else 0.0
    metadata = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "script": str(script_path),
        "device": _device_name(),
        "git_commit": _git_commit(script_path),
        "pytorch_version": getattr(torch, "__version__", "unknown"),
        "torch_npu_version": getattr(getattr(torch, "npu", None), "__version__", "unknown"),
        "cann_version": os.environ.get("CANN_VERSION", os.environ.get("ASCEND_VERSION", "unknown")),
        "warmup": warmup,
        "repeat": repeat,
        "timer": timer,
        "timer_source": {
            "device": "torch_npu.profiler",
            "host": "perf_counter_ns",
            "both": "torch_npu.profiler + perf_counter_ns",
        }[timer],
        "kernel_counts": kernel_counts,
        "device_min_us": min(device_samples) if device_samples else None,
        "device_median_us": statistics.median(device_samples) if device_samples else None,
        "host_device_min_us": min(host_device_samples) if host_device_samples else None,
        "host_device_median_us": statistics.median(host_device_samples) if host_device_samples else None,
        "error_type": type(failure).__name__ if failure else "",
        "error_message": str(failure) if failure else "",
        "traceback": failure_traceback,
    }
    metadata.update(config.get("metadata") or {})
    meta_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
    result = {
        "case_id": case_id,
        "case_name": case_name,
        "timer": timer,
        "device_us": device_us,
        "host_device_us": host_device_us,
        "status": status,
        "meta": meta_text,
        "script_path": script_path,
        "baseline_row": {
            "Case ID": case_id,
            "Device Time (us)": "{:.3f}".format(device_us) if device_samples else "",
            "Host + Device Time (us)": "{:.3f}".format(host_device_us) if host_device_samples else "",
            "Meta": meta_text,
        },
    }
    path = _write_result(result, config, mode, flag, threshold_pct)
    if not config.get("quiet", False) and path is not None:
        if timer == "both":
            measured = "device={:.3f} us host={:.3f} us".format(device_us, host_device_us)
        else:
            measured_us = device_us if timer == "device" else host_device_us
            measured = "{}={:.3f} us".format(timer, measured_us)
        print("[OpsPerf] {} [{}] {} -> {}".format(case_id, status, measured, path))
    if failure is not None:
        raise failure
    return result_value


def _finalize_pairs():
    with _PAIR_LOCK:
        pending = list(_PAIR_RESULTS.values())
        _PAIR_RESULTS.clear()
    for pair in pending:
        for result, config, _ in pair.values():
            row = {
                "Case ID": result["case_id"],
                "Meta": result["meta"],
            }
            if result["timer"] in ("device", "both"):
                row["Device Verdict"] = "INCOMPLETE_PAIR"
            if result["timer"] in ("host", "both"):
                row["Host Verdict"] = "INCOMPLETE_PAIR"
            _CsvWriter.append(
                result["script_path"],
                config.get("output_dir"),
                "performance",
                row,
                config.get("details", False),
                result["timer"],
            )


atexit.register(_finalize_pairs)
