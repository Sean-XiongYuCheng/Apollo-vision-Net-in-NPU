# LeRobot PI0.5 Ascend NPU 推理优化

本目录提供基于官方 LeRobot 精确基线制作的 PI0.5 独立推理优化补丁，以及 eager 和 TorchAir 双图时延测试脚本。

## 1. 包内容

~~~text
inference/
├── README.md
├── lerobot_pi05_npu_inference_b74a551.patch
└── test/
    └── pi05_latency.py
~~~

## 2. 官方基线

| 项目 | 值 |
| --- | --- |
| 仓库 | https://github.com/huggingface/lerobot.git |
| 来源分支 | main |
| 提交 | b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85 |
| LeRobot 版本 | 0.5.2 |

补丁只保证应用到该精确提交。不要在其他版本上跳过 git apply --check 强制安装。

## 3. 优化范围

补丁修改以下文件：

~~~text
src/lerobot/policies/factory.py
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py
src/lerobot/policies/pi05/vision_siglip_npu.py
src/lerobot/policies/pi_gemma.py
~~~

主要优化：

- 在 policy 移动到最终设备后提供统一推理准备入口。
- 融合 Gemma 和 SigLIP QKV 权重。
- 提供 PI0.5 独立 SigLIP NPU 视觉塔和融合 QKV，并固定使用 FP16 Q/K/V 执行 PFA，输出随后恢复视觉塔 dtype。
- 图像缩放直接保留 NCHW/NHWC 原布局，由 resize_with_pad_torch 识别格式，避免 NPU 上重复转置。
- 提供 AdaRMS NPU 融合与预计算 modulation 路径，并对齐 eager 条件张量与 fp32 权重 dtype。
- 预计算固定 10 步 denoise 查找表。
- 支持 eager NPU 融合和 TorchAir prefix/denoise 双图。

PI0.5 补丁不依赖 PI0 目录中的文件。

## 4. 环境准备

要求 Ascend NPU、匹配的驱动、CANN、Torch 与 Torch-NPU，并使用 Python 3.12 环境。

~~~bash
conda create -n lerobot-npu python=3.12 -y
conda activate lerobot-npu

python -m pip install --upgrade pip setuptools wheel
python -m pip install /path/to/torch-wheel.whl
python -m pip install /path/to/torch_npu-wheel.whl
~~~

不同驱动和 CANN 版本必须选择对应的官方 Torch-NPU wheel。

## 5. 拉取基线

~~~bash
export WORKSPACE=/path/to/workspace
export LEROBOT_REPO=$WORKSPACE/lerobot
export PATCH_DIR=/path/to/inference

cd "$WORKSPACE"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git lerobot
cd "$LEROBOT_REPO"
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85

test "$(git rev-parse HEAD)" = "b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85"
~~~

本流程通过 GIT_LFS_SKIP_SMUDGE=1 明确跳过官方 LeRobot 仓库的 Git LFS 大文件下载。仓库中的 LFS 文件主要是 tests/artifacts 下的测试图片、视频和数据样本，不参与本补丁的 PI0.5 推理。PI0.5 模型 checkpoint 不在这些仓库测试资产中，仍需在运行时通过本地目录或 Hub 模型 ID 单独提供。

只有需要运行依赖官方测试资产的上游测试时，才需要在仓库内额外执行 git lfs pull。

## 6. 校验并应用补丁

~~~bash
cd "$LEROBOT_REPO"
git apply --check "$PATCH_DIR/lerobot_pi05_npu_inference_b74a551.patch"
git apply --index "$PATCH_DIR/lerobot_pi05_npu_inference_b74a551.patch"
git diff --cached --check
~~~

补丁文件随 DrivingSDK 的 Git 版本统一管理，文件完整性由 Git 对象哈希和提交记录保障，因此不额外维护 `SHA256SUMS`。`git apply --check` 仍用于确认补丁与指定 LeRobot 基线兼容。

撤销尚未提交的补丁：

~~~bash
git reset --hard b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85
~~~

该命令会删除未提交修改，执行前应确认工作区没有其他内容。

## 7. 安装到 Conda 环境

~~~bash
conda activate lerobot-npu
source /path/to/Ascend/cann/set_env.sh

cd "$LEROBOT_REPO"
python -m pip install -e ".[pi]"
~~~

可编辑安装会把依赖安装到当前 Conda 环境，并让 lerobot 指向当前仓库的 src 目录，不需要额外设置 PYTHONPATH。每个新终端仍需执行 CANN 的 set_env.sh。

验证导入：

~~~bash
python - <<'PY'
import torch
import torch_npu
import tbe
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.vision_siglip_npu import PI05SiglipVisionModel

print("PI05Policy:", PI05Policy)
print("PI05SiglipVisionModel:", PI05SiglipVisionModel)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
print("TBE:", tbe.__file__)
PY
~~~

## 8. 准备模型

~~~bash
export PI05_MODEL=/path/to/pi05-checkpoint
export NPU_DEVICE=npu:0

test -f "$PI05_MODEL/config.json"
test -f "$PI05_MODEL/model.safetensors"
~~~

使用 Hub 模型 ID 时去掉测试命令中的 --local-files-only。

## 9. eager NPU 融合测试

完成第 7 节的可编辑安装后，可直接使用本目录中的测试脚本：

### 测试输入说明

测试脚本加载指定 checkpoint 的真实配置和模型权重，但用于计时的模型输入是合成数据：

- 图像由 `torch.rand` 随机生成；
- 文本 token 固定为全 1，attention mask 固定为全有效；
- state-as-text 的 processor、自然语言 tokenizer 和真实状态提示词路径未被执行；
- flow-matching 初始噪声由模型内部随机生成。

因此，该脚本只用于验证 PI0.5 推理路径及测量合成 batch 下的时延，不能用于评估模型精度、动作正确性或任务成功率。输出中的 `e2e` 仅覆盖合成 batch 进入 policy 后的图像预处理、动作采样和结果返回，不包含真实相机采集、自然语言分词、状态提示词构造、环境交互及机器人执行。

~~~bash
cd "$LEROBOT_REPO"

python "$PATCH_DIR/test/pi05_latency.py" \
  --pretrained "$PI05_MODEL" \
  --device "$NPU_DEVICE" \
  --warmup 2 \
  --iterations 10 \
  --text-length 32 \
  --local-files-only
~~~

测试脚本会关闭 checkpoint 中可能保留的通用 compile_model，再显式启用 QKV、视觉塔 PFA 和 eager NPU 融合路径。

脚本会在终端输出结果，并同步写入 $LEROBOT_REPO/output/YYYYMMDD_HHMMSS_pi05_inference.log。输出目录不存在时会自动创建；eager 与 TorchAir 测试使用相同命名规则。

## 10. TorchAir 双图测试

~~~bash
python "$PATCH_DIR/test/pi05_latency.py" \
  --pretrained "$PI05_MODEL" \
  --device "$NPU_DEVICE" \
  --graph-compile \
  --warmup 2 \
  --iterations 10 \
  --text-length 32 \
  --local-files-only
~~~

--graph-compile 会额外启用 prefix prefill 图和固定 10 步 denoise 图。

输出字段：

- e2e：包含策略图像预处理和模型动作采样。
- model.sample_actions：使用已经整理好的模型输入。
- p90：使用 int((N - 1) * 0.9) 对应的下界索引，不会在默认 10 次采样时退化为最大值。

## 11. 与 PI0 补丁组合

PI0 与 PI0.5 补丁修改的源码文件互不重叠，可以分别安装，也可以在同一官方基线上按任意顺序依次执行 git apply --index。

## 12. 常见问题

### 找不到 tbe

~~~bash
conda activate lerobot-npu
source /path/to/Ascend/cann/set_env.sh
python -c "import tbe; print(tbe.__file__)"
~~~

### 出现 Inductor 或 Triton driver 错误

确认使用本目录提供的测试脚本。脚本会关闭通用 compile_model，并由 --graph-compile 明确选择是否进入 TorchAir。

### 首次运行较慢

首次权重加载和 TorchAir 图编译属于一次性开销。正式统计只应使用预热后的同步 NPU 推理。
