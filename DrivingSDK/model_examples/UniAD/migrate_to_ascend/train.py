# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""UniAD NPU training entry (single entry for full+perf)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import runpy
import sys

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:  # 已有子进程创建后不可再修改，保持现状
    pass

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401,E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(ROOT, "UniAD")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from mx_driving.patcher import default_patcher  # noqa: E402
from migrate_to_ascend.patch import configure_patcher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--performance",
        action="store_true",
        help="stop early at UNIAD_BRAKE_AT (default 500, aligned with baseline perf.py)",
    )

    args, unknown = parser.parse_known_args()

    configure_patcher(default_patcher, performance=args.performance)
    default_patcher.apply()

    # Hand every upstream argument (config path, --launcher, --cfg-options,
    # --work-dir, ...) over to the unmodified tools/train.py entry.
    sys.argv = [os.path.abspath(__file__)] + unknown
    runpy.run_module("tools.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
