# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""PrecisionDebugger integration helpers for patcher training-loop utilities."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union

from mx_driving.patcher.patcher_logger import patcher_logger


_TASKS = {"statistics", "tensor", "acc_check", "nan_check", "structure", "diff_check", "exception_dump"}
_LEVELS = {"L0", "L1", "L2", "mix", "debug"}
_PRECISIONS = {"high", "low"}
_RISK_LEVELS = {"ALL", "CORE", "FOCUS"}
_DATA_MODES = {"all", "forward", "backward", "input", "output"}
_SUMMARY_MODES = {"statistics", "md5", "xor"}
_RANGE_RE = re.compile(r"^\d+-\d+$")
_COMMON_CONFIG_KEYS = {
    "task",
    "dump_path",
    "rank",
    "step",
    "level",
    "async_dump",
    "dump_enable",
    "extra_info",
    "precision",
    "risk_level",
    "custom_op_namespaces",
}
_TASK_SECTION_KEYS = {
    "scope",
    "list",
    "api_list",
    "tensor_list",
    "data_mode",
    "summary_mode",
    "diff_nums",
    "bench_path",
    "white_list",
    "black_list",
    "error_data_path",
}


class PrecisionDebuggerConfigError(ValueError):
    """Raised when PrecisionDebugger config is invalid before msprobe starts."""


def _check_optional_bool(config: Dict[str, Any], key: str) -> None:
    if key in config and not isinstance(config[key], bool):
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config field '{key}' must be a bool")


def _check_optional_choice(config: Dict[str, Any], key: str, choices: set) -> None:
    if key in config and config[key] not in choices:
        expected = ", ".join(sorted(choices))
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config field '{key}' must be one of: {expected}")


def _check_optional_str_list(config: Dict[str, Any], key: str) -> None:
    if key in config and (not isinstance(config[key], list) or any(not isinstance(item, str) for item in config[key])):
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config field '{key}' must be a list of strings")


def _check_rank_or_step(config: Dict[str, Any], key: str) -> None:
    if key not in config:
        return
    value = config[key]
    if not isinstance(value, list):
        raise PrecisionDebuggerConfigError(
            f"PrecisionDebugger config {key} must be a list of non-negative integers or range strings"
        )
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            continue
        if isinstance(item, str) and _RANGE_RE.match(item):
            start, end = item.split("-", 1)
            if int(start) <= int(end):
                continue
        raise PrecisionDebuggerConfigError(
            f"PrecisionDebugger config {key} must be a list of non-negative integers or range strings"
        )


def _check_data_mode(section: Dict[str, Any]) -> None:
    if "data_mode" not in section:
        return
    value = section["data_mode"]
    if not isinstance(value, list) or any(mode not in _DATA_MODES for mode in value):
        raise PrecisionDebuggerConfigError(
            "PrecisionDebugger config data_mode must be a list containing all, forward, backward, input, or output"
        )


def _check_summary_mode(section: Dict[str, Any]) -> None:
    if "summary_mode" not in section:
        return
    value = section["summary_mode"]
    if isinstance(value, str):
        if value not in _SUMMARY_MODES:
            raise PrecisionDebuggerConfigError("PrecisionDebugger config summary_mode must be statistics, md5, or xor")
        return
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return
    raise PrecisionDebuggerConfigError("PrecisionDebugger config summary_mode must be a string or list of strings")


def _check_task_section(config: Dict[str, Any]) -> None:
    task = config.get("task", "statistics")
    section = config.get(task)
    if section is not None and not isinstance(section, dict):
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config section '{task}' must be an object")

    if isinstance(section, dict):
        _check_optional_str_list(section, "scope")
        _check_optional_str_list(section, "list")
        _check_optional_str_list(section, "tensor_list")
        _check_optional_str_list(section, "white_list")
        _check_optional_str_list(section, "black_list")
        _check_data_mode(section)
        _check_summary_mode(section)

    if config.get("async_dump") is True and task == "tensor":
        tensor_section = config.get("tensor")
        tensor_list = tensor_section.get("list") if isinstance(tensor_section, dict) else None
        if not isinstance(tensor_list, list) or not tensor_list:
            raise PrecisionDebuggerConfigError(
                "PrecisionDebugger config tensor async_dump requires tensor.list to limit dumped tensors"
            )
        if isinstance(tensor_section, dict) and tensor_section.get("summary_mode") == "md5":
            raise PrecisionDebuggerConfigError("PrecisionDebugger config async_dump does not support summary_mode=md5")


def _validate_precision_debugger_config_object(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise PrecisionDebuggerConfigError("PrecisionDebugger config root must be a JSON object")
    if "dump_path" not in config:
        raise PrecisionDebuggerConfigError("PrecisionDebugger config missing required field: dump_path")
    if not isinstance(config["dump_path"], str):
        raise PrecisionDebuggerConfigError("PrecisionDebugger config field 'dump_path' must be a string")

    _check_optional_choice(config, "task", _TASKS)
    _check_optional_choice(config, "level", _LEVELS)
    _check_optional_choice(config, "precision", _PRECISIONS)
    _check_optional_choice(config, "risk_level", _RISK_LEVELS)
    _check_optional_bool(config, "async_dump")
    _check_optional_bool(config, "dump_enable")
    _check_optional_bool(config, "extra_info")
    _check_optional_str_list(config, "custom_op_namespaces")
    _check_rank_or_step(config, "rank")
    _check_rank_or_step(config, "step")
    _check_task_section(config)
    return config


def validate_precision_debugger_config(config_path: Union[os.PathLike, str]) -> Tuple[str, Dict[str, Any]]:
    """Validate the stable subset of msprobe config.json used by patcher."""
    normalized_path = os.fspath(config_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"PrecisionDebugger config file not found: {normalized_path}")
    if not os.access(normalized_path, os.R_OK):
        raise PermissionError(f"PrecisionDebugger config file is not readable: {normalized_path}")

    try:
        with open(normalized_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise PrecisionDebuggerConfigError(
            f"PrecisionDebugger config file is not valid JSON: {normalized_path}"
        ) from exc

    return normalized_path, _validate_precision_debugger_config_object(config)


def build_precision_debugger_config(**kwargs: Any) -> Dict[str, Any]:
    """Build an msprobe config object from Patcher keyword arguments."""
    config = dict(kwargs)
    if not config.get("dump_path"):
        raise PrecisionDebuggerConfigError(
            "PrecisionDebugger requires config_path or dump_path when using direct parameters"
        )

    task = config.get("task", "statistics")
    if task not in _TASKS:
        expected = ", ".join(sorted(_TASKS))
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config field 'task' must be one of: {expected}")

    if "api_list" in config:
        if "list" in config:
            raise PrecisionDebuggerConfigError("PrecisionDebugger config cannot set both 'api_list' and 'list'")
        config["list"] = config.pop("api_list")

    explicit_section = config.pop(task, {})
    if explicit_section is None:
        explicit_section = {}
    if not isinstance(explicit_section, dict):
        raise PrecisionDebuggerConfigError(f"PrecisionDebugger config section '{task}' must be an object")

    section = dict(explicit_section)
    for key in sorted(_TASK_SECTION_KEYS - {"api_list"}):
        if key in config:
            section[key] = config.pop(key)

    unknown_keys = sorted(key for key in config if key not in _COMMON_CONFIG_KEYS and key not in _TASKS)
    if unknown_keys:
        raise PrecisionDebuggerConfigError(
            "PrecisionDebugger config contains unsupported direct parameter(s): " + ", ".join(unknown_keys)
        )

    config.setdefault("task", task)
    config.setdefault("rank", [])
    config.setdefault("step", [])
    config.setdefault("level", "L1")
    config.setdefault("async_dump", False)
    config.setdefault("extra_info", True)
    if section:
        config[task] = section

    return _validate_precision_debugger_config_object(config)


class PrecisionDebuggerController:
    """Lazy wrapper around msprobe.pytorch.PrecisionDebugger."""

    def __init__(
        self,
        config_path: Optional[Union[os.PathLike, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        seed_mode: bool = True,
        debugger_factory: Optional[Callable[..., Any]] = None,
        seed_all_func: Optional[Callable[..., Any]] = None,
    ):
        if config_path is not None and config is not None:
            raise PrecisionDebuggerConfigError("PrecisionDebugger accepts either config_path or direct config")
        if config_path is None and config is None:
            raise PrecisionDebuggerConfigError("PrecisionDebugger requires config_path or direct config")
        self.config_path = None
        self._owns_config_path = False
        if config_path is not None:
            self.config_path, self.config = validate_precision_debugger_config(config_path)
        else:
            self.config = _validate_precision_debugger_config_object(dict(config))
        self.seed = seed
        self.seed_mode = seed_mode
        self._debugger_factory = debugger_factory
        self._seed_all_func = seed_all_func
        self._debugger = None
        self._seeded = False
        self._started = False

    @staticmethod
    def _load_msprobe():
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError as exc:
            raise ImportError("PrecisionDebugger requires msprobe. Install mindstudio-probe and retry.") from exc
        return PrecisionDebugger, seed_all

    def _ensure_config_path(self) -> str:
        if self.config_path is not None:
            return self.config_path

        config_fd, config_path = tempfile.mkstemp(prefix="mx_driving_precision_debugger_", suffix=".json")
        try:
            with os.fdopen(config_fd, "w", encoding="utf-8") as config_file:
                json.dump(self.config, config_file, ensure_ascii=False, indent=2)
        except Exception:
            try:
                os.remove(config_path)
            except OSError:
                pass
            raise
        self.config_path = config_path
        self._owns_config_path = True
        return config_path

    def _cleanup_generated_config(self) -> None:
        if not self._owns_config_path or self.config_path is None:
            return
        try:
            os.remove(self.config_path)
        except OSError as exc:  # pragma: no cover - defensive path
            patcher_logger.warning(f"PrecisionDebugger generated config cleanup failed: {exc}")
        finally:
            self.config_path = None
            self._owns_config_path = False

    def _resolve_msprobe(self) -> Tuple[Any, Callable[..., Any]]:
        """Resolve PrecisionDebugger and seed_all, importing msprobe at most once."""
        precision_debugger_cls = self._debugger_factory
        seed_all_func = self._seed_all_func
        if precision_debugger_cls is None or seed_all_func is None:
            imported_debugger_cls, imported_seed_all = self._load_msprobe()
            precision_debugger_cls = precision_debugger_cls or imported_debugger_cls
            seed_all_func = seed_all_func or imported_seed_all
        return precision_debugger_cls, seed_all_func

    def seed_all(self) -> None:
        """Fix randomness, only effective before model and dataloader construction."""
        if self.seed is None or self._seeded:
            return
        _, seed_all_func = self._resolve_msprobe()
        seed_all_func(seed=self.seed, mode=self.seed_mode)
        self._seeded = True

    def initialize(self) -> None:
        """Import msprobe and instantiate PrecisionDebugger.

        msprobe wraps its supported torch API list while PrecisionDebugger is being
        constructed, so this has to run after the other patches are applied and before
        model code binds direct references such as `from torch import matmul`.
        """
        self._ensure_debugger()

    def _ensure_debugger(self):
        if self._debugger is not None:
            return self._debugger

        precision_debugger_cls, _ = self._resolve_msprobe()
        self.seed_all()

        config_path = self._ensure_config_path()
        try:
            self._debugger = precision_debugger_cls(config_path=config_path)
        except Exception as exc:
            self._cleanup_generated_config()
            raise RuntimeError(f"Failed to initialize PrecisionDebugger with config_path={config_path}: {exc}") from exc
        return self._debugger

    def start(self, model: Optional[Any] = None) -> None:
        debugger = self._ensure_debugger()
        if model is None:
            debugger.start()
        else:
            debugger.start(model)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._ensure_debugger().stop()
        self._started = False

    def step(self) -> None:
        self._ensure_debugger().step()

    def _stop_and_step_quietly(self) -> None:
        for action_name, action in (("stop", self.stop), ("step", self.step)):
            try:
                action()
            except Exception as exc:  # pragma: no cover - defensive path
                patcher_logger.warning(f"PrecisionDebugger {action_name} failed during cleanup: {exc}")

    def _cleanup_after_success(self) -> None:
        self.stop()
        self.step()

    def run_step(self, func: Callable[[], Any], model: Optional[Any] = None) -> Any:
        """Run one training iteration between PrecisionDebugger start/stop/step."""
        self.start(model)
        try:
            result = func()
        except BaseException:
            self._stop_and_step_quietly()
            raise
        self._cleanup_after_success()
        return result

    def finish(self) -> None:
        """Finish one training loop, keeping the debugger usable for later loops."""
        if self._started:
            self._stop_and_step_quietly()

    def close(self) -> None:
        """Release the debugger. Called by whoever owns this controller."""
        self.finish()

        debugger = self._debugger
        if debugger is None:
            self._cleanup_generated_config()
            return
        for method_name in ("close", "exit"):
            method = getattr(debugger, method_name, None)
            if not callable(method):
                continue
            try:
                method()  # pylint: disable=not-callable
            except Exception as exc:  # pragma: no cover - defensive path
                patcher_logger.warning(f"PrecisionDebugger {method_name} failed during cleanup: {exc}")
            break
        self._debugger = None
        self._cleanup_generated_config()


def build_precision_debugger_controller(options: Dict[str, Any]) -> Optional[PrecisionDebuggerController]:
    """Build a controller from training-loop patch options."""
    if not options.get("enable_precision_debugger"):
        return None
    return PrecisionDebuggerController(
        config_path=options.get("precision_debugger_config_path"),
        config=options.get("precision_debugger_config"),
        seed=options.get("precision_debugger_seed"),
        seed_mode=options.get("precision_debugger_seed_mode", True),
    )
