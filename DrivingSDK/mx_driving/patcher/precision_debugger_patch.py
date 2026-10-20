# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Minimal framework adapters for PrecisionDebugger training-step boundaries."""

import importlib
from functools import wraps
from typing import Dict

from mx_driving.patcher.patch import LegacyPatch


def build_mmengine_precision_debugger_patch(options: Dict) -> LegacyPatch:
    """Wrap MMEngine train-loop ``run_iter`` methods without replacing the loops."""

    def _apply(module, _options):
        controller = options["precision_debugger_controller"]
        importlib.import_module(f"{module.__name__}.runner")

        def wrap_run_iter(loop_cls):
            original = loop_cls.run_iter

            @wraps(original)
            def run_iter(self, *args, **kwargs):
                model = getattr(getattr(self, "runner", None), "model", None)
                return controller.run_step(
                    lambda: original(self, *args, **kwargs),
                    model=model,
                )

            loop_cls.run_iter = run_iter

        wrap_run_iter(module.runner.EpochBasedTrainLoop)
        wrap_run_iter(module.runner.IterBasedTrainLoop)

    return LegacyPatch(_apply, target_module="mmengine")


def build_mmcv_precision_debugger_patch(options: Dict) -> LegacyPatch:
    """Bracket MMCV train-iteration hooks while leaving runner loops unchanged.

    MMCV 1.x performs backward and optimizer updates in ``after_train_iter``
    hooks, so wrapping only ``run_iter`` would stop collection too early.
    """

    def _apply(module, _options):
        controller = options["precision_debugger_controller"]
        importlib.import_module(f"{module.__name__}.runner")

        def patch_runner(runner_cls):
            original_call_hook = runner_cls.call_hook
            original_run_iter = runner_cls.run_iter

            @wraps(original_call_hook)
            def call_hook(self, fn_name, *args, **kwargs):
                if fn_name == "before_train_iter":
                    controller.start(getattr(self, "model", None))
                    try:
                        return original_call_hook(self, fn_name, *args, **kwargs)
                    except BaseException:
                        controller.finish()
                        raise

                if fn_name == "after_train_iter":
                    try:
                        result = original_call_hook(self, fn_name, *args, **kwargs)
                    except BaseException:
                        controller.finish()
                        raise
                    controller.stop()
                    controller.step()
                    return result

                return original_call_hook(self, fn_name, *args, **kwargs)

            @wraps(original_run_iter)
            def run_iter(self, *args, **kwargs):
                try:
                    return original_run_iter(self, *args, **kwargs)
                except BaseException:
                    controller.finish()
                    raise

            runner_cls.call_hook = call_hook
            runner_cls.run_iter = run_iter

        patch_runner(module.runner.EpochBasedRunner)
        patch_runner(module.runner.IterBasedRunner)

    return LegacyPatch(_apply, target_module="mmcv")
