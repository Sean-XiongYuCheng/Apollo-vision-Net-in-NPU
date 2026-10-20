# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
PrecisionDebugger patcher utility tests.

These tests use fake msprobe objects and temporary config files so they do not
require NPU hardware or mindstudio-probe to be installed.
"""

# Test doubles intentionally receive framework-managed attributes after construction.
# pylint: disable=attribute-defined-outside-init

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest


_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_patcher_dir = os.path.join(_project_root, "mx_driving", "patcher")


def _load_module_from_file(module_name: str, file_path: str):
    """Load a module directly from file without triggering parent package __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_patcher_logger = _load_module_from_file(
    "mx_driving.patcher.patcher_logger", os.path.join(_patcher_dir, "patcher_logger.py")
)
_reporting = _load_module_from_file("mx_driving.patcher.reporting", os.path.join(_patcher_dir, "reporting.py"))
_version_module = _load_module_from_file("mx_driving.patcher.version", os.path.join(_patcher_dir, "version.py"))
_patch_module = _load_module_from_file("mx_driving.patcher.patch", os.path.join(_patcher_dir, "patch.py"))
_precision_debugger_module = _load_module_from_file(
    "mx_driving.patcher.precision_debugger", os.path.join(_patcher_dir, "precision_debugger.py")
)
_patcher_module = _load_module_from_file("mx_driving.patcher.patcher", os.path.join(_patcher_dir, "patcher.py"))
_legacy_module = _load_module_from_file("mx_driving.patcher.legacy", os.path.join(_patcher_dir, "legacy.py"))
_precision_debugger_patch_module = _load_module_from_file(
    "mx_driving.patcher.precision_debugger_patch",
    os.path.join(_patcher_dir, "precision_debugger_patch.py"),
)


PrecisionDebuggerConfigError = _precision_debugger_module.PrecisionDebuggerConfigError
PrecisionDebuggerController = _precision_debugger_module.PrecisionDebuggerController
build_precision_debugger_config = _precision_debugger_module.build_precision_debugger_config
validate_precision_debugger_config = _precision_debugger_module.validate_precision_debugger_config
Patcher = _patcher_module.Patcher
LegacyPatcherBuilder = _legacy_module.LegacyPatcherBuilder
build_mmengine_precision_debugger_patch = _precision_debugger_patch_module.build_mmengine_precision_debugger_patch
build_mmcv_precision_debugger_patch = _precision_debugger_patch_module.build_mmcv_precision_debugger_patch


def _write_config(directory: str, config: dict) -> str:
    path = os.path.join(directory, "precision_debugger_config.json")
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    return path


def _minimal_config(dump_path: str = "./msprobe_dump") -> dict:
    return {
        "task": "statistics",
        "dump_path": dump_path,
        "rank": [],
        "step": [],
        "level": "L1",
        "async_dump": False,
        "extra_info": True,
        "statistics": {
            "scope": [],
            "list": [],
            "tensor_list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics",
        },
    }


class TestPrecisionDebuggerConfigValidation(unittest.TestCase):
    """Test stable config validation before msprobe is invoked."""

    def test_validate_minimal_statistics_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_config(tmpdir, _minimal_config())
            normalized_path, config = validate_precision_debugger_config(path)

        self.assertEqual(normalized_path, path)
        self.assertEqual(config["task"], "statistics")
        self.assertEqual(config["dump_path"], "./msprobe_dump")

    def test_missing_dump_path_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_config(tmpdir, {"task": "statistics"})
            with self.assertRaisesRegex(PrecisionDebuggerConfigError, "dump_path"):
                validate_precision_debugger_config(path)

    def test_rank_and_step_accept_ranges_and_reject_invalid_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_config = _minimal_config()
            valid_config["rank"] = [0, "1-3"]
            valid_config["step"] = [0, 2, "4-6"]
            validate_precision_debugger_config(_write_config(tmpdir, valid_config))

            invalid_config = _minimal_config()
            invalid_config["rank"] = ["3-1"]
            path = _write_config(tmpdir, invalid_config)
            with self.assertRaisesRegex(PrecisionDebuggerConfigError, "rank"):
                validate_precision_debugger_config(path)

    def test_rank_and_step_reject_booleans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for key in ("rank", "step"):
                invalid_config = _minimal_config()
                invalid_config[key] = [True]
                path = _write_config(tmpdir, invalid_config)
                with self.assertRaisesRegex(PrecisionDebuggerConfigError, key):
                    validate_precision_debugger_config(path)

    def test_async_tensor_dump_requires_limited_tensor_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "task": "tensor",
                "dump_path": "./msprobe_dump",
                "level": "L1",
                "async_dump": True,
                "tensor": {
                    "scope": [],
                    "list": [],
                    "data_mode": ["all"],
                    "summary_mode": "statistics",
                },
            }
            with self.assertRaisesRegex(PrecisionDebuggerConfigError, "tensor.list"):
                validate_precision_debugger_config(_write_config(tmpdir, config))

    def test_summary_mode_is_checked_for_common_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_config()
            config["statistics"]["summary_mode"] = "bad_mode"
            with self.assertRaisesRegex(PrecisionDebuggerConfigError, "summary_mode"):
                validate_precision_debugger_config(_write_config(tmpdir, config))

    def test_build_direct_config_defaults_and_task_section(self):
        config = build_precision_debugger_config(
            dump_path="./direct_dump",
            task="tensor",
            level="mix",
            step=[0],
            rank=[0],
            api_list=["Module.backbone.Conv2d.forward.0"],
            data_mode=["all"],
        )

        self.assertEqual(config["dump_path"], "./direct_dump")
        self.assertEqual(config["task"], "tensor")
        self.assertEqual(config["level"], "mix")
        self.assertEqual(config["rank"], [0])
        self.assertEqual(config["step"], [0])
        self.assertEqual(config["tensor"]["list"], ["Module.backbone.Conv2d.forward.0"])
        self.assertEqual(config["tensor"]["data_mode"], ["all"])

    def test_build_direct_config_requires_dump_path(self):
        with self.assertRaisesRegex(PrecisionDebuggerConfigError, "dump_path"):
            build_precision_debugger_config(task="statistics")


class TestPrecisionDebuggerController(unittest.TestCase):
    """Test PrecisionDebugger lifecycle wrapper with fake msprobe objects."""

    def test_run_step_calls_seed_init_start_stop_step_in_order(self):
        events = []

        class FakePrecisionDebugger:
            def __init__(self, config_path):
                events.append(("init", os.path.basename(config_path)))

            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

            def step(self):
                events.append("step")

        def fake_seed_all(seed, mode):
            events.append(("seed", seed, mode))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            controller = PrecisionDebuggerController(
                config_path,
                seed=1234,
                seed_mode=False,
                debugger_factory=FakePrecisionDebugger,
                seed_all_func=fake_seed_all,
            )
            result = controller.run_step(lambda: events.append("train") or "ok")

        self.assertEqual(result, "ok")
        self.assertEqual(
            events,
            [
                ("seed", 1234, False),
                ("init", "precision_debugger_config.json"),
                "start",
                "train",
                "stop",
                "step",
            ],
        )

    def test_run_step_preserves_training_exception_and_cleans_up(self):
        events = []

        class FakePrecisionDebugger:
            def __init__(self, config_path):
                del config_path

            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

            def step(self):
                events.append("step")

        def fail_training():
            events.append("train")
            raise RuntimeError("training failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            controller = PrecisionDebuggerController(
                config_path,
                debugger_factory=FakePrecisionDebugger,
                seed_all_func=lambda **kwargs: None,
            )
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                controller.run_step(fail_training)

        self.assertEqual(events, ["start", "train", "stop", "step"])

    def test_run_step_passes_model_to_start_when_provided(self):
        events = []
        model = object()

        class FakePrecisionDebugger:
            def __init__(self, config_path):
                del config_path

            def start(self, start_model):
                events.append(("start", start_model))

            def stop(self):
                events.append("stop")

            def step(self):
                events.append("step")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            controller = PrecisionDebuggerController(
                config_path,
                debugger_factory=FakePrecisionDebugger,
                seed_all_func=lambda **kwargs: None,
            )
            controller.run_step(lambda: events.append("train"), model=model)

        self.assertEqual(events, [("start", model), "train", "stop", "step"])

    def test_direct_config_creates_temporary_config_and_cleans_it(self):
        events = []
        seen_paths = []

        class FakePrecisionDebugger:
            def __init__(self, config_path):
                seen_paths.append(config_path)
                with open(config_path, "r", encoding="utf-8") as config_file:
                    events.append(("init", json.load(config_file)["dump_path"]))

            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

            def step(self):
                events.append("step")

            def close(self):
                events.append("close")

        controller = PrecisionDebuggerController(
            config=build_precision_debugger_config(dump_path="./direct_dump", step=[0]),
            debugger_factory=FakePrecisionDebugger,
            seed_all_func=lambda **kwargs: None,
        )
        controller.run_step(lambda: events.append("train"))
        self.assertTrue(os.path.exists(seen_paths[0]))
        controller.close()

        self.assertEqual(events, [("init", "./direct_dump"), "start", "train", "stop", "step", "close"])
        self.assertFalse(os.path.exists(seen_paths[0]))


class TestPatcherPrecisionDebuggerConfiguration(unittest.TestCase):
    """Test Patcher and legacy builder configuration entry points."""

    def test_patcher_with_precision_debugger_sets_options_and_chains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            patcher = Patcher()
            result = patcher.with_precision_debugger(config_path, seed=7, seed_mode=False)

        self.assertIs(result, patcher)
        self.assertIsNotNone(patcher._precision_debugger_options)
        self.assertTrue(patcher._precision_debugger_options["enable_precision_debugger"])
        self.assertEqual(patcher._precision_debugger_options["precision_debugger_seed"], 7)
        self.assertFalse(patcher._precision_debugger_options["precision_debugger_seed_mode"])

    def test_patcher_with_precision_debugger_rejects_missing_config(self):
        patcher = Patcher()
        with self.assertRaises(FileNotFoundError):
            patcher.with_precision_debugger("/path/does/not/exist.json")

    def test_patcher_with_precision_debugger_accepts_direct_parameters(self):
        patcher = Patcher()
        result = patcher.with_precision_debugger(
            dump_path="./direct_dump",
            task="statistics",
            level="L1",
            step=[0],
            rank=[],
        )

        self.assertIs(result, patcher)
        options = patcher._precision_debugger_options
        self.assertTrue(options["enable_precision_debugger"])
        self.assertNotIn("precision_debugger_config_path", options)
        self.assertEqual(options["precision_debugger_config"]["dump_path"], "./direct_dump")
        self.assertEqual(options["precision_debugger_config"]["step"], [0])

    def test_patcher_with_precision_debugger_rejects_mixed_config_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            patcher = Patcher()
            with self.assertRaisesRegex(PrecisionDebuggerConfigError, "cannot be combined"):
                patcher.with_precision_debugger(config_path, dump_path="./override")

    def test_legacy_builder_forwards_precision_debugger_to_new_patcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, _minimal_config())
            builder = LegacyPatcherBuilder()
            builder.with_precision_debugger(config_path, seed=5)
            patcher_package = types.ModuleType("mx_driving.patcher")
            patcher_package._LEGACY_NAME_TO_CLASS = {}
            original_package = sys.modules.get("mx_driving.patcher")
            sys.modules["mx_driving.patcher"] = patcher_package
            try:
                legacy_patcher = builder.build()
            finally:
                if original_package is None:
                    del sys.modules["mx_driving.patcher"]
                else:
                    sys.modules["mx_driving.patcher"] = original_package

        options = legacy_patcher._patcher._precision_debugger_options
        self.assertIsNotNone(options)
        self.assertEqual(options["precision_debugger_seed"], 5)
        self.assertEqual(options["precision_debugger_config_path"], config_path)

    def test_legacy_builder_forwards_direct_precision_debugger_parameters(self):
        builder = LegacyPatcherBuilder()
        builder.with_precision_debugger(dump_path="./direct_dump", step=[0])
        patcher_package = types.ModuleType("mx_driving.patcher")
        patcher_package._LEGACY_NAME_TO_CLASS = {}
        original_package = sys.modules.get("mx_driving.patcher")
        sys.modules["mx_driving.patcher"] = patcher_package
        try:
            legacy_patcher = builder.build()
        finally:
            if original_package is None:
                del sys.modules["mx_driving.patcher"]
            else:
                sys.modules["mx_driving.patcher"] = original_package

        options = legacy_patcher._patcher._precision_debugger_options
        self.assertIsNotNone(options)
        self.assertEqual(options["precision_debugger_config"]["dump_path"], "./direct_dump")
        self.assertEqual(options["precision_debugger_config"]["step"], [0])


class TestPatcherPrecisionDebuggerInitialization(unittest.TestCase):
    """Test when Patcher seeds and instantiates PrecisionDebugger during apply()."""

    @staticmethod
    def _fake_controller(events, initialize_error=None):
        class FakePrecisionDebuggerController:
            def seed_all(self):
                events.append("seed_all")

            def initialize(self):
                if initialize_error is not None:
                    raise initialize_error
                events.append("initialize")

            def close(self):
                events.append("close")

        return FakePrecisionDebuggerController()

    def _patch_controller_factory(self, controller):
        original = _precision_debugger_module.build_precision_debugger_controller
        _precision_debugger_module.build_precision_debugger_controller = lambda options: controller
        self.addCleanup(setattr, _precision_debugger_module, "build_precision_debugger_controller", original)

    def _make_patcher(self, controller):
        self._patch_controller_factory(controller)
        patcher = Patcher()
        patcher.with_precision_debugger(dump_path="./msprobe_dump", step=[0])
        return patcher

    def test_apply_seeds_before_patches_and_initializes_after(self):
        events = []
        controller = self._fake_controller(events)
        patcher = self._make_patcher(controller)

        target_module = types.ModuleType("fake_precision_debugger_target")
        target_module.op = lambda: None
        sys.modules["fake_precision_debugger_target"] = target_module
        self.addCleanup(sys.modules.pop, "fake_precision_debugger_target", None)

        def record_patch(original):
            events.append("patch")
            return original

        patcher.add(_patch_module.AtomicPatch("fake_precision_debugger_target.op", target_wrapper=record_patch))
        patcher.apply()

        self.assertEqual(events, ["seed_all", "patch", "initialize"])
        self.assertIs(patcher.precision_debugger, controller)

    def test_apply_propagates_initialization_failure(self):
        events = []
        controller = self._fake_controller(events, initialize_error=ImportError("msprobe not installed"))
        patcher = self._make_patcher(controller)

        with self.assertRaisesRegex(ImportError, "msprobe not installed"):
            patcher.apply()


class TestPrecisionDebuggerTrainingLoopPatches(unittest.TestCase):
    """Unit-test the small adapters without replacing framework training loops."""

    @staticmethod
    def _install_fake_module(module_name, module):
        originals = {}
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                originals[name] = sys.modules[name]
                del sys.modules[name]
        sys.modules[module_name] = module
        return originals

    @staticmethod
    def _restore_modules(module_name, originals):
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                del sys.modules[name]
        sys.modules.update(originals)

    @staticmethod
    def _fake_controller(events):
        class FakePrecisionDebuggerController:
            def __init__(self):
                self.started = False

            def start(self, model=None):
                events.append(("pd_start", model))
                self.started = True

            def stop(self):
                if self.started:
                    events.append("pd_stop")
                    self.started = False

            def step(self):
                events.append("pd_step")

            def run_step(self, func, model=None):
                self.start(model)
                try:
                    result = func()
                except BaseException:
                    self.finish()
                    raise
                self.stop()
                self.step()
                return result

            def finish(self):
                if self.started:
                    self.stop()
                    self.step()

        return FakePrecisionDebuggerController()

    def test_mmengine_wraps_run_iter_without_replacing_loop_methods(self):
        events = []
        controller = self._fake_controller(events)

        class EpochBasedTrainLoop:
            def run_iter(self, idx, data_batch):
                events.append(("epoch_run_iter", idx, data_batch))

        class IterBasedTrainLoop:
            def run_iter(self, data_batch):
                events.append(("iter_run_iter", data_batch))

        runner_module = types.ModuleType("mmengine.runner")
        runner_module.EpochBasedTrainLoop = EpochBasedTrainLoop
        runner_module.IterBasedTrainLoop = IterBasedTrainLoop
        mmengine_module = types.ModuleType("mmengine")
        mmengine_module.runner = runner_module
        originals = self._install_fake_module("mmengine", mmengine_module)
        sys.modules["mmengine.runner"] = runner_module

        try:
            options = {"enable_precision_debugger": True, "precision_debugger_controller": controller}
            patch = build_mmengine_precision_debugger_patch(options)
            result = patch.apply()
            self.assertEqual(result.status.value, "applied")

            model = object()
            epoch_loop = EpochBasedTrainLoop()
            epoch_loop.runner = types.SimpleNamespace(model=model)
            epoch_loop.run_iter(3, "epoch_batch")

            iter_loop = IterBasedTrainLoop()
            iter_loop.runner = types.SimpleNamespace(model=model)
            iter_loop.run_iter("iter_batch")
        finally:
            self._restore_modules("mmengine", originals)

        self.assertEqual(
            events,
            [
                ("pd_start", model),
                ("epoch_run_iter", 3, "epoch_batch"),
                "pd_stop",
                "pd_step",
                ("pd_start", model),
                ("iter_run_iter", "iter_batch"),
                "pd_stop",
                "pd_step",
            ],
        )

    def test_mmcv_brackets_train_hooks_and_leaves_validation_untouched(self):
        events = []
        controller = self._fake_controller(events)

        class Runner:
            def __init__(self):
                self.model = object()

            def call_hook(self, name):
                events.append(("hook", name))
                if name == "after_train_iter":
                    events.append("optimizer_step")

            def run_iter(self, data_batch, train_mode):
                events.append(("run_iter", data_batch, train_mode))

        class EpochBasedRunner(Runner):
            pass

        class IterBasedRunner(Runner):
            pass

        runner_module = types.ModuleType("mmcv.runner")
        runner_module.EpochBasedRunner = EpochBasedRunner
        runner_module.IterBasedRunner = IterBasedRunner
        mmcv_module = types.ModuleType("mmcv")
        mmcv_module.runner = runner_module

        originals = self._install_fake_module("mmcv", mmcv_module)
        sys.modules["mmcv.runner"] = runner_module

        try:
            options = {"enable_precision_debugger": True, "precision_debugger_controller": controller}
            patch = build_mmcv_precision_debugger_patch(options)
            result = patch.apply()
            self.assertEqual(result.status.value, "applied")

            runner = EpochBasedRunner()
            runner.call_hook("before_train_iter")
            runner.run_iter("train_batch", train_mode=True)
            runner.call_hook("after_train_iter")
            runner.call_hook("before_val_iter")
            runner.run_iter("val_batch", train_mode=False)
            runner.call_hook("after_val_iter")
        finally:
            self._restore_modules("mmcv", originals)

        self.assertEqual(
            events,
            [
                ("pd_start", runner.model),
                ("hook", "before_train_iter"),
                ("run_iter", "train_batch", True),
                ("hook", "after_train_iter"),
                "optimizer_step",
                "pd_stop",
                "pd_step",
                ("hook", "before_val_iter"),
                ("run_iter", "val_batch", False),
                ("hook", "after_val_iter"),
            ],
        )

    def test_mmcv_cleans_up_when_run_iter_fails(self):
        events = []
        controller = self._fake_controller(events)

        class Runner:
            model = object()

            def call_hook(self, name):
                events.append(("hook", name))

            def run_iter(self, *_args, **_kwargs):
                events.append("run_iter")
                raise RuntimeError("training failed")

        class EpochBasedRunner(Runner):
            pass

        class IterBasedRunner(Runner):
            pass

        runner_module = types.ModuleType("mmcv.runner")
        runner_module.EpochBasedRunner = EpochBasedRunner
        runner_module.IterBasedRunner = IterBasedRunner
        mmcv_module = types.ModuleType("mmcv")
        mmcv_module.runner = runner_module
        originals = self._install_fake_module("mmcv", mmcv_module)
        sys.modules["mmcv.runner"] = runner_module

        try:
            options = {"enable_precision_debugger": True, "precision_debugger_controller": controller}
            patch = build_mmcv_precision_debugger_patch(options)
            self.assertEqual(patch.apply().status.value, "applied")

            runner = IterBasedRunner()
            runner.call_hook("before_train_iter")
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                runner.run_iter("batch", train_mode=True)
        finally:
            self._restore_modules("mmcv", originals)

        self.assertEqual(
            events,
            [
                ("pd_start", runner.model),
                ("hook", "before_train_iter"),
                "run_iter",
                "pd_stop",
                "pd_step",
            ],
        )


if __name__ == "__main__":
    unittest.main()
