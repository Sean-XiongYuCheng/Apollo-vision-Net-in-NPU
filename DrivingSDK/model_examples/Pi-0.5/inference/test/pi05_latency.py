#!/usr/bin/env python
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / "src" / "lerobot").is_dir():
    sys.path.insert(0, str(REPO_ROOT / "src"))


class TeeStream:
    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def resolve_lerobot_root() -> Path:
    candidates = []
    if configured_root := os.environ.get("LEROBOT_REPO"):
        candidates.append(Path(configured_root))
    candidates.extend([Path.cwd(), REPO_ROOT])

    spec = importlib.util.find_spec("lerobot")
    if spec is not None and spec.origin is not None:
        candidates.append(Path(spec.origin).resolve().parents[2])

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "src" / "lerobot").is_dir():
            return resolved
    raise RuntimeError("Cannot locate LeRobot root; set LEROBOT_REPO or run from the repository root")


def create_log_path(lerobot_root: Path) -> Path:
    output_dir = lerobot_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{timestamp}_pi05_inference.log"


def synchronize(device) -> None:
    import torch

    # NPU 算子默认异步执行，测量前后同步才能覆盖实际设备执行时间。
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int((len(values) - 1) * fraction)]


def report(name: str, values: list[float]) -> None:
    print(
        f"{name}: mean={statistics.mean(values):.3f} ms "
        f"median={statistics.median(values):.3f} ms p90={percentile(values, 0.9):.3f} ms"
    )


def resolve_device(value: str | None):
    import torch

    if value:
        return torch.device(value)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure PI0.5 NPU inference paths.")
    parser.add_argument("--pretrained", required=True, help="Local model directory or Hub model id.")
    parser.add_argument("--device", default=None, help="Torch device, for example npu:0.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--graph-compile", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations < 1 or args.text_length < 1:
        raise ValueError("warmup must be non-negative; iterations and text-length must be positive")

    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device = resolve_device(args.device)
    config = PreTrainedConfig.from_pretrained(args.pretrained, local_files_only=args.local_files_only)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config, got {type(config).__name__}")
    config.device = str(device)
    # checkpoint 可能保留训练时的通用 torch.compile 配置；时延脚本只启用下方显式选择的推理路径。
    config.compile_model = False
    config.gradient_checkpointing = False
    config.compile_inference_graph = args.graph_compile
    # PI0.5 支持在 eager 融合路径和 TorchAir 双图路径之间切换。
    policy = PI05Policy.from_pretrained(
        args.pretrained,
        config=config,
        local_files_only=args.local_files_only,
    )
    policy.prepare_inference_optimizations(
        enable_npu_fused_ops=True,
        enable_graph_compile=args.graph_compile,
        enable_qkv_fusion=True,
    )
    policy.eval()

    image_key = next(iter(policy.config.image_features))
    height, width = policy.config.image_resolution
    batch_size = 1
    batch = {
        image_key: torch.rand(batch_size, 3, height, width, device=device),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, args.text_length, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, args.text_length, dtype=torch.bool, device=device),
    }
    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    def end_to_end() -> torch.Tensor:
        return policy.predict_action_chunk(batch)

    def model_path() -> torch.Tensor:
        return policy.model.sample_actions(images, image_masks, tokens, masks)

    for _ in range(args.warmup):
        # 预热用于触发首次图编译和运行时缓存，避免初始化开销污染统计结果。
        end_to_end()
        model_path()
    synchronize(device)

    e2e_values = []
    model_values = []
    for _ in range(args.iterations):
        # 完整路径包含策略输入整理、视觉处理、KV cache 构造和动作采样。
        synchronize(device)
        start = time.perf_counter()
        output = end_to_end()
        synchronize(device)
        e2e_values.append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        output = model_path()
        synchronize(device)
        model_values.append((time.perf_counter() - start) * 1000.0)

    print(f"model=pi05 device={device} output_shape={tuple(output.shape)}")
    report("e2e", e2e_values)
    report("model.sample_actions", model_values)


def main() -> int:
    args = parse_args()
    log_path = create_log_path(resolve_lerobot_root())
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(TeeStream(sys.stdout, log_file)), redirect_stderr(TeeStream(sys.stderr, log_file)):
            print(f"log_file={log_path}")
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
