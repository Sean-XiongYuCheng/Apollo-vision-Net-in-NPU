#!/bin/bash
# DiffusionDriveV2 PDMS 精度验证脚本, 使用官方 run_pdm_score_fast.py 命令。
#
# 支持 --ckpt 指定评估权重 (默认 ckpts/diffusiondrivev2_sel.ckpt), 默认评估子集navtest
#
# 用法:
#   bash migrate_to_ascend/eval_pdm_score.sh                          # 默认官方权重 ckpts/diffusiondrivev2_sel.ckpt
#   bash migrate_to_ascend/eval_pdm_score.sh --ckpt exp/xxx/last.ckpt # 评估指定权重
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

CKPT=${CKPT:-ckpts/diffusiondrivev2_sel.ckpt}
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT=$2; shift 2 ;;
        --help|-h)
            cat <<'HELP'
Usage:
  bash migrate_to_ascend/eval_pdm_score.sh                            # default ckpt: ckpts/diffusiondrivev2_sel.ckpt
  bash migrate_to_ascend/eval_pdm_score.sh --ckpt exp/xxx/last.ckpt   # evaluate a specific ckpt

Runs the official run_pdm_score_fast.py on navtest and reports the best-of-N PDMS.
Split and cache paths are fixed in this script.

Options:
  --ckpt PATH        checkpoint to evaluate   (default ckpts/diffusiondrivev2_sel.ckpt)
  --help, -h         show this help

Outputs:
  migrate_to_ascend/output/full/eval_pdm_navtest.log
HELP
            exit 0 ;;
        *)
            echo "[EVAL] Unknown option: $1 (only --ckpt is supported)" >&2
            echo "[EVAL] Run 'bash migrate_to_ascend/eval_pdm_score.sh --help' for details." >&2
            exit 2 ;;
    esac
done

[ -f "${CKPT}" ] || { echo "[EVAL] 权重不存在: ${CKPT}" >&2; exit 1; }

# 评估日志统一写 OUT_DIR (默认 output/full; train_full.sh 会 export 同一目录, 便于报告解析)。
OUT_DIR=${OUT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output/full"}
EVAL_LOG="${OUT_DIR}/eval_pdm_navtest.log"
mkdir -p "${OUT_DIR}"
# 开跑前打印权重与日志路径, 便于实时 tail 评估日志。
echo "[EVAL] 评估权重: ${CKPT} (navtest 全量) | 日志: ${EVAL_LOG}"

# 加载昇腾运行环境 (数据/缓存路径与运行时参数)。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_npu.sh"

PYTHON=${PYTHON:-python}
${PYTHON} "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_fast.py" \
    agent=diffusiondrivev2_sel_agent \
    experiment_name=diffusiondrivev2_agent_eval \
    train_test_split=navtest \
    agent.checkpoint_path="${CKPT}" \
    +metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/" \
    +test_cache_path="${NAVSIM_EXP_ROOT}/metric_feature_cache/" > "${EVAL_LOG}" 2>&1
RC=$?

if [ ${RC} -eq 0 ]; then
    # 汇总 best-of-N PDMS 及各分量, 追加到日志尾部 (train_full.sh 的报告直接解析本行)。
    SUMMARY=$(${PYTHON} - "${EVAL_LOG}" <<'PYEOF'
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
d = {}
for k, v in re.findall(r"\s+(val/\S+)\s+([0-9.eE+-]+)", txt):
    try:
        d[k] = float(v)
    except ValueError:
        pass

IDX = (0, 1, 2)

SCORE = ("val/fine_reward_%d_epoch", "val/fine_%d_final_epoch")

def cand_score(i):
    for pat in SCORE:
        if pat % i in d:
            return d[pat % i]
    return None

avail = [i for i in IDX if cand_score(i) is not None]
b = max(avail, key=cand_score) if avail else None
best = cand_score(b) if b is not None else None

frags = []
frags.append("pdms=%.4f" % best if best is not None else "pdms=N/A")
frags.append("best_index=%s" % (b if b is not None else "N/A"))
for s in ("no_collision", "drivable_area", "ttc", "progress", "comfort", "dir_weighted", "prog_raw"):
    k = "val/fine_%s_%s_epoch" % (b, s)
    if b is not None and k in d:
        frags.append("%s=%.4f" % (s, d[k]))
print(" ".join(frags))
PYEOF
)
    echo "===== [PDMS-SUMMARY] ${SUMMARY} =====" >> "${EVAL_LOG}"
    # 流水线内 (train_full.sh 设了 EVAL_QUIET) 由精度报告统一输出 PDMS, 避免同一份结果打印两次。
    if [ -z "${EVAL_QUIET:-}" ]; then
        echo "[EVAL] PDMS (best-of-N): ${SUMMARY}"
    fi
    if [[ "${SUMMARY}" == *"pdms=N/A"* ]]; then
        echo "[EVAL] 警告: 日志中未找到 val/fine_reward_*_epoch 或 val/fine_*_final_epoch, 请检查 ${EVAL_LOG}" >&2
    fi
fi
exit ${RC}
