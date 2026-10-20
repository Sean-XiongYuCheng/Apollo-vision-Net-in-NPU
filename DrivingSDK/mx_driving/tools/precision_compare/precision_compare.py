"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

precision_compare 工具使用指南请见 docs/zh/tools/precision_compare.md
"""

import numpy as np
from collections import namedtuple

CompareResult = namedtuple(
    'CompareResult',
    [
        'passed',
        'pass_rate',
        'max_abs_err',
        'max_rel_err',
        'mean_abs_err',
        'mean_rel_err',
        'error_indices',
        'error_benchmarks',
        'error_predictions',
        'error_abs_diffs',
        'error_rel_diffs',
        'nan_count_pred',
        'nan_count_bench',
        'inf_count_pred',
        'inf_count_bench',
        'total_count',
    ],
)

_INNER_WIDTH = 67
_TOP_BORDER = '+' + '=' * (_INNER_WIDTH + 2) + '+'
_BOTTOM_BORDER = _TOP_BORDER
_SECTION_DIV = '+' + '-' * (_INNER_WIDTH + 2) + '+'
_ROW_DIV = '|' + '-' * (_INNER_WIDTH + 2) + '|'


def _content_line(text):
    return '| ' + text.ljust(_INNER_WIDTH) + ' |'


def _to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    try:
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(data, (list, tuple)):
        return np.array(data)
    if isinstance(data, (int, float)):
        return np.array([data])
    raise TypeError(f"Unsupported data type: {type(data)}, expected numpy.ndarray, torch.Tensor, list, or scalar")


def _to_float64(arr):
    if arr.dtype == np.float16 or (hasattr(np, 'float32') and arr.dtype == np.float32):
        return arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64)
    if arr.dtype != np.float64:
        try:
            return arr.astype(np.float64)
        except (ValueError, TypeError):
            pass
    return arr


def _is_integer_dtype(arr):
    return np.issubdtype(arr.dtype, np.integer)


def _compute_errors(pred, bench, err_threshold, abs_threshold, is_integer):
    nan_pred = np.isnan(pred)
    nan_bench = np.isnan(bench)
    both_nan = nan_pred & nan_bench
    one_nan = nan_pred ^ nan_bench
    any_nan = nan_pred | nan_bench

    pos_inf_pred = np.isposinf(pred)
    pos_inf_bench = np.isposinf(bench)
    neg_inf_pred = np.isneginf(pred)
    neg_inf_bench = np.isneginf(bench)
    both_same_inf = (pos_inf_pred & pos_inf_bench) | (neg_inf_pred & neg_inf_bench)
    any_inf = np.isinf(pred) | np.isinf(bench)
    special_mask = any_nan | any_inf

    safe_pred = np.where(special_mask, 0.0, pred)
    safe_bench = np.where(special_mask, 0.0, bench)
    abs_diff = np.abs(np.subtract(safe_pred, safe_bench))

    if is_integer:
        rel_diff = abs_diff.copy()
        point_passed = abs_diff == 0
        point_passed[one_nan] = False
        abs_diff[one_nan] = np.inf
        rel_diff[one_nan] = np.inf
        return abs_diff, rel_diff, point_passed

    epsilon = max(err_threshold, 1e-10)
    denominator = np.maximum(np.abs(safe_pred), np.abs(safe_bench))
    denominator = np.maximum(denominator, epsilon)
    rel_diff = np.where(abs_diff < err_threshold, abs_diff, abs_diff / denominator)

    point_passed = (abs_diff <= abs_threshold) | (rel_diff <= err_threshold)
    point_passed = point_passed | both_nan | both_same_inf
    point_passed[one_nan] = False

    abs_diff[both_nan | both_same_inf] = 0.0
    rel_diff[both_nan | both_same_inf] = 0.0
    abs_diff[one_nan] = np.inf
    rel_diff[one_nan] = np.inf
    abs_diff[(~one_nan) & any_inf & (~both_same_inf)] = np.inf
    rel_diff[(~one_nan) & any_inf & (~both_same_inf)] = np.inf

    return abs_diff, rel_diff, point_passed


def _print_detail(pred_flat, bench_flat, abs_diff, rel_diff, display_lines):
    total = pred_flat.size
    half = display_lines // 2

    print(_SECTION_DIV)
    print(_content_line(f'{"Loop":>8s}  {"ExpectOut":>12s}  {"RealOut":>12s}  {"FpDiff":>12s}  {"RateDiff":>12s}'))
    print(_ROW_DIV)

    if total <= display_lines:
        for i in range(total):
            line = f'{i + 1:08d}  {bench_flat[i]:12.7f}  {pred_flat[i]:12.7f}  {abs_diff[i]:12.7f}  {rel_diff[i]:12.7f}'
            print(_content_line(line))
    else:
        for i in range(half):
            line = f'{i + 1:08d}  {bench_flat[i]:12.7f}  {pred_flat[i]:12.7f}  {abs_diff[i]:12.7f}  {rel_diff[i]:12.7f}'
            print(_content_line(line))
        print(_content_line(f'{"...":>8s}  {"...":>12s}  {"...":>12s}  {"...":>12s}  {"...":>12s}'))
        for i in range(total - half, total):
            line = f'{i + 1:08d}  {bench_flat[i]:12.7f}  {pred_flat[i]:12.7f}  {abs_diff[i]:12.7f}  {rel_diff[i]:12.7f}'
            print(_content_line(line))


def _print_errors(pred_flat, bench_flat, abs_diff, rel_diff, err_indices, display_lines):
    if err_indices.size == 0:
        return
    num_errors = err_indices.size
    half = display_lines // 2

    max_rel = np.max(rel_diff[err_indices])
    print(_ROW_DIV)
    print(_content_line(f'Maximum relative error is: {max_rel:.7f}.'))
    print(_content_line('Error Line'))
    print(_ROW_DIV)
    print(_content_line(f'{"Loop":>8s}  {"ExpectOut":>12s}  {"RealOut":>12s}  {"FpDiff":>12s}  {"RateDiff":>12s}'))
    print(_ROW_DIV)

    if num_errors <= display_lines:
        for idx in err_indices:
            line = f'{idx + 1:08d}  {bench_flat[idx]:12.7f}  {pred_flat[idx]:12.7f}  {abs_diff[idx]:12.7f}  {rel_diff[idx]:12.7f}'
            print(_content_line(line))
    else:
        for idx in err_indices[:half]:
            line = f'{idx + 1:08d}  {bench_flat[idx]:12.7f}  {pred_flat[idx]:12.7f}  {abs_diff[idx]:12.7f}  {rel_diff[idx]:12.7f}'
            print(_content_line(line))
        print(_content_line(f'{"...":>8s}  {"...":>12s}  {"...":>12s}  {"...":>12s}  {"...":>12s}'))
        for idx in err_indices[-half:]:
            line = f'{idx + 1:08d}  {bench_flat[idx]:12.7f}  {pred_flat[idx]:12.7f}  {abs_diff[idx]:12.7f}  {rel_diff[idx]:12.7f}'
            print(_content_line(line))


def _infer_default_threshold(pred, bench):
    """根据输入dtype推断默认的 err_threshold 和 err_ratio。"""
    is_int = _is_integer_dtype(_to_numpy(pred)) and _is_integer_dtype(_to_numpy(bench))
    if is_int:
        return 0.0, 0.0
    dtype = _to_numpy(pred).dtype
    if dtype == np.float16:
        return 1e-3, 1e-3
    return 1e-4, 1e-4


def precision_compare(
    npu_pred,
    benchmark,
    err_threshold=None,
    err_ratio=None,
    *,
    abs_threshold=None,
    max_diff_hard=0.1,
    display_lines=30,
    verbose=True,
):
    pred = _to_float64(_to_numpy(npu_pred).ravel())
    bench = _to_float64(_to_numpy(benchmark).ravel())

    if pred.shape != bench.shape:
        raise ValueError(f"Shape mismatch: npu_pred {npu_pred.shape} vs benchmark {benchmark.shape}")
    if pred.size == 0:
        raise ValueError("Input data is empty")

    if err_threshold is None or err_ratio is None:
        inferred_thr, inferred_ratio = _infer_default_threshold(npu_pred, benchmark)
        if err_threshold is None:
            err_threshold = inferred_thr
        if err_ratio is None:
            err_ratio = inferred_ratio

    is_integer = _is_integer_dtype(_to_numpy(npu_pred)) and _is_integer_dtype(_to_numpy(benchmark))

    if err_threshold < 0:
        raise ValueError(f"err_threshold must be >= 0, got {err_threshold}")
    if err_threshold == 0 and not is_integer:
        raise ValueError("err_threshold=0 is only allowed for integer dtypes, got float input")
    if not (0 <= err_ratio <= 1):
        raise ValueError(f"err_ratio must be in [0, 1], got {err_ratio}")
    if display_lines < 2:
        raise ValueError(f"display_lines must be >= 2, got {display_lines}")

    if abs_threshold is None:
        abs_threshold = err_threshold

    nan_count_pred = int(np.isnan(pred).sum())
    nan_count_bench = int(np.isnan(bench).sum())
    inf_count_pred = int(np.isinf(pred).sum())
    inf_count_bench = int(np.isinf(bench).sum())

    abs_diff, rel_diff, point_passed = _compute_errors(pred, bench, err_threshold, abs_threshold, is_integer)

    total_count = pred.size
    pass_count = int(point_passed.sum())
    pass_rate = pass_count / total_count * 100.0
    err_indices = np.where(~point_passed)[0]

    finite_mask = np.isfinite(abs_diff) & np.isfinite(rel_diff)
    max_abs_err = float(np.max(abs_diff[finite_mask])) if np.any(finite_mask) else 0.0
    max_rel_err = float(np.max(rel_diff[finite_mask])) if np.any(finite_mask) else 0.0
    mean_abs_err = float(np.mean(abs_diff[finite_mask])) if np.any(finite_mask) else 0.0
    mean_rel_err = float(np.mean(rel_diff[finite_mask])) if np.any(finite_mask) else 0.0

    required_pass_rate = (1.0 - err_ratio) * 100.0
    passed = pass_rate >= required_pass_rate
    if not is_integer and max_rel_err >= max_diff_hard:
        passed = False

    if verbose:
        print()
        print(_TOP_BORDER)
        print(_content_line(f'total_count:{total_count};'))

        has_special = nan_count_pred + nan_count_bench + inf_count_pred + inf_count_bench > 0
        if has_special:
            warn = (
                f"Warning: npu_pred contains {nan_count_pred} NaN, {inf_count_pred} Inf; "
                f"benchmark contains {nan_count_bench} NaN, {inf_count_bench} Inf."
            )
            print(_SECTION_DIV)
            print(_content_line(warn))

        _print_detail(pred, bench, abs_diff, rel_diff, display_lines)

        print(_SECTION_DIV)
        print(_content_line(f'{"DiffThd":>8s}  {"PctThd":>12s}  {"PctRlt":>12s}  {"Result":>8s}'))
        print(_ROW_DIV)
        result_str = "Pass" if passed else "Failed"
        summary = f'{err_threshold:8.4f}  {required_pass_rate:11.2f}%  {pass_rate:11.6f}%  {result_str:>8s}'
        print(_content_line(summary))

        if err_indices.size > 0 and not passed:
            _print_errors(pred, bench, abs_diff, rel_diff, err_indices, display_lines)

        print(_BOTTOM_BORDER)

    error_benchmarks = bench[err_indices] if err_indices.size > 0 else np.array([])
    error_predictions = pred[err_indices] if err_indices.size > 0 else np.array([])
    error_abs_diffs = abs_diff[err_indices] if err_indices.size > 0 else np.array([])
    error_rel_diffs = rel_diff[err_indices] if err_indices.size > 0 else np.array([])

    return CompareResult(
        passed=passed,
        pass_rate=pass_rate,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        mean_abs_err=mean_abs_err,
        mean_rel_err=mean_rel_err,
        error_indices=err_indices,
        error_benchmarks=error_benchmarks,
        error_predictions=error_predictions,
        error_abs_diffs=error_abs_diffs,
        error_rel_diffs=error_rel_diffs,
        nan_count_pred=nan_count_pred,
        nan_count_bench=nan_count_bench,
        inf_count_pred=inf_count_pred,
        inf_count_bench=inf_count_bench,
        total_count=total_count,
    )


def assert_precision(npu_pred, benchmark, err_threshold=None, err_ratio=None, **kwargs):
    res = precision_compare(npu_pred, benchmark, err_threshold, err_ratio, **kwargs)
    return res.passed


def check_precision(npu_pred, benchmark, err_threshold=None, err_ratio=None, **kwargs):
    res = precision_compare(npu_pred, benchmark, err_threshold, err_ratio, **kwargs)
    if not res.passed:
        raise AssertionError(
            f"Precision check failed: pass_rate={res.pass_rate:.4f}%, max_rel_err={res.max_rel_err:.7f}"
        )
