#!/bin/bash
# DiffusionDriveV2 全量精度训练和评估脚本 (严格链式: 本次 RL -> 本次 SEL -> navtest PDMS 评估)。
#
# 用法:
#   bash migrate_to_ascend/train_full.sh                                    # 全量 + 自动评估
#
# 参数 (默认与官方对齐):
#   --rl_epochs N     RL 阶段 epochs   (默认 10)
#   --sel_epochs N    SEL 阶段 epochs  (默认 20)
#   --rank_size N     使用的 NPU 卡数  (默认 8)
#   --batch_size N    每卡 batch size  (默认 64)
#
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根

RL_EPOCHS=10; SEL_EPOCHS=20; RANK_SIZE=8; BATCH_SIZE=64

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            cat <<'HELP'
Usage:
  bash migrate_to_ascend/train_full.sh                                   # full run + auto eval

Full accuracy pipeline: RL (10 ep) -> SEL (20 ep, init from this run's RL ckpt) -> navtest PDMS eval.

Options:
  --rl_epochs N      RL stage epochs            (default 10)
  --sel_epochs N     SEL stage epochs           (default 20)
  --rank_size N      number of NPU devices      (default 8)
  --batch_size N     per-device batch size      (default 64)
  --help, -h         show this help

Outputs:
  migrate_to_ascend/output/full/accuracy_report.txt
  migrate_to_ascend/output/full/{rl.log,sel.log,eval_pdm_navtest.log,rl_stage.ckpt,sel_stage.ckpt}
HELP
            exit 0 ;;
        --rl_epochs)  RL_EPOCHS=$2; shift 2 ;;
        --sel_epochs) SEL_EPOCHS=$2; shift 2 ;;
        --rank_size)  RANK_SIZE=$2; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash migrate_to_ascend/train_full.sh [--rl_epochs N] [--sel_epochs N] [--rank_size N] [--batch_size N]" >&2
            echo "Run 'bash migrate_to_ascend/train_full.sh --help' for details." >&2
            exit 1 ;;
    esac
done

# 精度流程日志/报告目录 (export 后 train.sh / eval_pdm_score.sh 子脚本共用同一目录)。
export OUT_DIR="$(pwd)/migrate_to_ascend/output/full"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/accuracy_report.txt"
echo "[ACC-TEST] DiffusionDriveV2 accuracy: RL ${RL_EPOCHS}ep + SEL ${SEL_EPOCHS}ep | ${RANK_SIZE} devices x ${BATCH_SIZE}/device"

# 加载昇腾运行环境 (训练与评估共用; export 会被子进程 train.sh 继承)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_npu.sh"

# 从阶段日志中取本次运行的产物目录: 官方 run_training.py 会打印，取主进程也就是第一条
#   Path where all results are stored: <NAVSIM_EXP_ROOT>/<experiment_name>/<时间戳>
run_dir_from_log() {
    local log=$1 d
    d=$(grep -a "Path where all results are stored:" "${log}" 2>/dev/null | head -1 | sed 's/^.*stored:[[:space:]]*//')
    [ -n "${d}" ] || return 1
    echo "${d}"
}

# 取产物目录内本次训练写出的权重: 按 mtime 取最新一个。官方默认回调在每个 epoch 末保存
# 用法: pick_ckpt <阶段名> <产物目录>
pick_ckpt() {
    local tag=$1 dir=$2 ck
    ck=$(ls -t "${dir}"/lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -z "${ck}" ] || [ ! -s "${ck}" ]; then
        echo "[ACC-TEST] ERROR: ${tag} 阶段未产出 checkpoint (${dir})" >&2
        return 1
    fi
    echo "${ck}"
}

# 官方 RL 初始权重检查 (RL 阶段从官方权重启动, 与 train_performance.sh 一致)。
if [ ! -e "ckpts/diffusiondrive_navsim_88p1_PDMS" ]; then
    echo "[ACC-TEST] ERROR: RL 初始权重不存在: ckpts/diffusiondrive_navsim_88p1_PDMS" >&2
    echo "[ACC-TEST]        请从 https://huggingface.co/hustvl/DiffusionDrive 下载后放入 ckpts/" >&2
    exit 1
fi

# 阶段失败不再立即退出: 记录 RC 并跳过后续阶段, 最后仍输出报告, 便于查看已成功阶段的指标。
RC=0
EVAL_RC=-1   # -1 表示评估未执行 (前序阶段失败)

# ---- 1. RL 训练 ----
bash migrate_to_ascend/train.sh --stage rl --rl_epochs "${RL_EPOCHS}" \
    --rank_size "${RANK_SIZE}" --batch_size "${BATCH_SIZE}"
RL_RC=$?
if [[ ${RL_RC} -ne 0 ]]; then
    RC=${RL_RC}   # 失败详情已由 train.sh 打印, 这里不重复
else
    RL_DIR=$(run_dir_from_log "${OUT_DIR}/rl.log") || {
        echo "[ACC-TEST] ERROR: 无法从 ${OUT_DIR}/rl.log 确定本次 RL 结果目录" >&2; RC=1; }
    if [[ ${RC} -eq 0 ]]; then
        RL_CKPT=$(pick_ckpt RL "${RL_DIR}") || RC=1
        if [[ ${RC} -eq 0 ]]; then
            # 拷贝成固定名字: 既作为交付产物, 也作为 SEL 的初始权重 (新路径不含 "=", 便于 Hydra 传参)。
            cp -f "${RL_CKPT}" "${OUT_DIR}/rl_stage.ckpt"
            echo "[ACC-TEST] RL done, 权重: ${OUT_DIR}/rl_stage.ckpt"
        fi
    fi
fi

# ---- 2. SEL 训练: 以本次 RL 权重 为初始权重 ----
if [[ ${RC} -eq 0 ]]; then
    CHECKPOINT_SEL="${OUT_DIR}/rl_stage.ckpt" bash migrate_to_ascend/train.sh --stage sel \
        --sel_epochs "${SEL_EPOCHS}" \
        --rank_size "${RANK_SIZE}" --batch_size "${BATCH_SIZE}"
    SEL_RC=$?
    if [[ ${SEL_RC} -ne 0 ]]; then
        RC=${SEL_RC}   # 失败详情已由 train.sh 打印, 这里不重复
    else
        SEL_DIR=$(run_dir_from_log "${OUT_DIR}/sel.log") || {
            echo "[ACC-TEST] ERROR: 无法从 ${OUT_DIR}/sel.log 确定本次 SEL 结果目录" >&2; RC=1; }
        if [[ ${RC} -eq 0 ]]; then
            SEL_CKPT=$(pick_ckpt SEL "${SEL_DIR}") || RC=1
            if [[ ${RC} -eq 0 ]]; then
                cp -f "${SEL_CKPT}" "${OUT_DIR}/sel_stage.ckpt"
                echo "[ACC-TEST] SEL done, 权重: ${OUT_DIR}/sel_stage.ckpt"
            fi
        fi
    fi
fi

# ---- 3. navtest 评估 (用拷贝后的 SEL 权重, 路径不含 "=") ----
EVAL_LOG="${OUT_DIR}/eval_pdm_navtest.log"
if [[ ${RC} -eq 0 ]]; then
    # EVAL_QUIET=1: 流水线内 PDMS 由下面的精度报告统一输出, eval 不再重复打印。
    EVAL_QUIET=1 bash migrate_to_ascend/eval_pdm_score.sh --ckpt "${OUT_DIR}/sel_stage.ckpt"
    EVAL_RC=$?
    [[ ${EVAL_RC} -ne 0 ]] && RC=${EVAL_RC}
fi

# ---- 4. 汇总精度报告 (解析 eval 输出中的 [PDMS-SUMMARY]) ----
declare -A MV
if [[ ${EVAL_RC} -eq 0 ]]; then
    _SUM=$(grep -a "\[PDMS-SUMMARY\]" "${EVAL_LOG}" | tail -1 | sed -E 's/^.*\[PDMS-SUMMARY\] //; s/ =====$//')
    for _kv in ${_SUM}; do
        _k=${_kv%%=*}; _v=${_kv#*=}
        [ -n "${_k}" ] && MV["${_k}"]="${_v}"
    done
fi

{
    echo "------------------ Final result ------------------"
    echo " DiffusionDriveV2 精度验证报告 | $(date '+%Y-%m-%d %H:%M:%S')"
    if [[ ${EVAL_RC} -eq -1 ]]; then
        echo "评估未执行 (前序训练阶段失败, 无 PDMS)"
    elif [[ ${EVAL_RC} -ne 0 ]]; then
        echo "评估失败 (rc=${EVAL_RC}), 详见 eval_pdm_navtest.log"
    else
        # 精度/性能指标统一为 "Key = Value" 风格 (与 performance_report.txt 一致)。
        echo "PDMS = ${MV[pdms]:-N/A}"
        echo "BestCandidateIndex = ${MV[best_index]:-N/A}"
        declare -A _MNAME=(
            [no_collision]=NoCollision [drivable_area]=DrivableArea [ttc]=TTC
            [progress]=Progress [comfort]=Comfort [dir_weighted]=DirWeighted [prog_raw]=ProgRaw
        )
        for _c in no_collision drivable_area ttc progress comfort dir_weighted prog_raw; do
            [ -n "${MV[${_c}]:-}" ] && echo "${_MNAME[${_c}]} = ${MV[${_c}]}"
        done
    fi
    # 只汇总实际跑过的阶段 (日志由 train.sh 生成), 未跑到的阶段不占篇幅。
    for _S in rl sel; do
        [ -f "${OUT_DIR}/${_S}.log" ] || continue
        echo "------------------------------------------------------------"
        echo "${_S^^} stage:"
        _PSUM=$(grep -a "PERF-SUMMARY" "${OUT_DIR}/${_S}.log" 2>/dev/null | tail -1)
        if [ -n "${_PSUM}" ]; then
            _E2E=$(echo "${_PSUM}"   | grep -oaE 'total_s=[0-9.]+' | head -1 | cut -d= -f2)
            _FPS=$(echo "${_PSUM}"   | grep -oaE 'fps=[0-9.]+'     | head -1 | cut -d= -f2)
            _AVG=$(echo "${_PSUM}"   | grep -oaE 'avg_s=[0-9.]+'   | head -1 | cut -d= -f2)
            _STEPS=$(echo "${_PSUM}" | grep -oaE 'n_steps=[0-9]+'  | head -1 | cut -d= -f2)
        else
            _E2E=""; _FPS=""; _AVG=""; _STEPS=""
        fi
        echo "  FPS = ${_FPS:-N/A}"
        echo "  E2ETrainingTime = ${_E2E:-N/A}"
        echo "  StepTime = ${_AVG:-N/A}"
        echo "  Steps = ${_STEPS:-N/A}"
    done
} > "${REPORT}"
cat "${REPORT}"
echo "[ACC-TEST] 精度报告: ${REPORT}"
exit ${RC}
