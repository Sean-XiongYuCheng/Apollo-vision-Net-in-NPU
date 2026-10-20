#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="${WORKSPACE_ROOT}/mmcv:${WORKSPACE_ROOT}/mmdetection_2.19:${WORKSPACE_ROOT}/mmsegmentation_0.20:${WORKSPACE_ROOT}/mmdetection3d_0.18.1:${WORKSPACE_ROOT}/Apollo-Vision-Net${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "===== Apollo NPU FP32 环境检查 ====="
echo "工作区: ${WORKSPACE_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "可见NPU: ${ASCEND_RT_VISIBLE_DEVICES:-未显式设置}"

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

print("Python版本:", sys.version.replace("\n", " "))

module_names = [
    "torch",
    "torch_npu",
    "mmcv",
    "mmdet",
    "mmseg",
    "mmdet3d",
    "mx_driving",
]

for name in module_names:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "无__version__")
    path = getattr(module, "__file__", "无__file__")
    print(f"{name}: version={version}, path={path}")

import torch
import torch_npu
import mmcv._ext

if not torch.npu.is_available():
    raise RuntimeError("torch.npu.is_available() 返回 False")

device = torch.device("npu:0")
value = torch.tensor([1.0, 2.0], device=device)
result = (value * 2).cpu().tolist()

if result != [2.0, 4.0]:
    raise RuntimeError(f"NPU计算结果异常: {result}")

print("mmcv._ext:", mmcv._ext.__file__)
print("NPU计算结果:", result)
print("APOLLO_NPU_ENV_CHECK=PASS")
PY
