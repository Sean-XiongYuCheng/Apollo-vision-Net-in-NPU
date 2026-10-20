#!/bin/bash
# =============================================================================
# DiffusionDriveV2 性能训练脚本。
# 默认 RL / SEL 阶段各跑 1 个 epoch, 均使用 ckpts/ 下官方初始权重启动
# (RL: diffusiondrive_navsim_88p1_PDMS, SEL: diffusiondrivev2_rl.ckpt,
# 训练结束后打印各阶段E2E 耗时 / 平均单步耗时 / FPS。
#
# 参数:
#   --stage S       运行阶段: all|rl|sel    (默认 all)
#   --rl_epochs N   RL 阶段 epochs           (默认 1)
#   --sel_epochs N  SEL 阶段 epochs          (默认 1)
#   --rank_size N   使用的 NPU 卡数          (默认 8)
#   --batch_size N  每卡 batch size          (默认 64)
#
# 用法:
#   bash migrate_to_ascend/train_performance.sh                                # RL/SEL 各 1 epoch
#   bash migrate_to_ascend/train_performance.sh --stage rl --rank_size 2       # 只跑 RL, 2 卡
#   bash migrate_to_ascend/train_performance.sh --batch_size 32 --rank_size 16  # 自定义 batch_size/卡数
#
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根

RL_EPOCHS=${RL_EPOCHS:-1}
SEL_EPOCHS=${SEL_EPOCHS:-1}
RANK_SIZE=${RANK_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
STAGE=${STAGE:-all}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            cat <<'HELP'
Usage:
  bash migrate_to_ascend/train_performance.sh                                 # RL/SEL each 1 epoch
  bash migrate_to_ascend/train_performance.sh --stage rl --rank_size 2        # RL only, 2 devices
  bash migrate_to_ascend/train_performance.sh --batch_size 32 --rank_size 16

Both stages start from the official weights in ckpts/ (no dependency on this run's products).

Options:
  --stage S          stage to run: all|rl|sel   (default all)
  --rl_epochs N      RL stage epochs            (default 1)
  --sel_epochs N     SEL stage epochs           (default 1)
  --rank_size N      number of NPU devices      (default 8)
  --batch_size N     per-device batch size      (default 64)
  --help, -h         show this help

Outputs:
  migrate_to_ascend/output/performance/performance_report.txt
  migrate_to_ascend/output/performance/{rl.log,sel.log}
HELP
            exit 0 ;;
        --stage)      STAGE=$2; shift 2 ;;
        --rl_epochs)  RL_EPOCHS=$2; shift 2 ;;
        --sel_epochs) SEL_EPOCHS=$2; shift 2 ;;
        --rank_size)  RANK_SIZE=$2; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash migrate_to_ascend/train_performance.sh [--stage all|rl|sel] [--rl_epochs N] [--sel_epochs N] [--rank_size N] [--batch_size N]" >&2
            echo "Run 'bash migrate_to_ascend/train_performance.sh --help' for details." >&2
            exit 1 ;;
    esac
done

case "${STAGE}" in all|rl|sel) ;; *) echo "ERROR: --stage 必须 all|rl|sel"; exit 1 ;; esac

# 性能流程日志/报告目录 (export 后子脚本 train.sh 写入同一目录, 与精度流程 output/full 隔离)。
export OUT_DIR="$(pwd)/migrate_to_ascend/output/performance"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
echo "[PERF-TEST] DiffusionDriveV2 performance: RL ${RL_EPOCHS}ep + SEL ${SEL_EPOCHS}ep | ${RANK_SIZE} devices x ${BATCH_SIZE}/device"

# 官方初始权重检查 (性能测试不依赖本次训练产物)。
case "${STAGE}" in
    all|rl)
        if [ ! -e "ckpts/diffusiondrive_navsim_88p1_PDMS" ]; then
            echo "[PERF-TEST] ERROR: RL 初始权重不存在: ckpts/diffusiondrive_navsim_88p1_PDMS" >&2
            echo "[PERF-TEST]        请从 https://huggingface.co/hustvl/DiffusionDrive 下载后放入 ckpts/" >&2
            exit 1
        fi ;;
esac
case "${STAGE}" in
    all|sel)
        if [ ! -f "ckpts/diffusiondrivev2_rl.ckpt" ]; then
            echo "[PERF-TEST] ERROR: SEL 初始权重不存在: ckpts/diffusiondrivev2_rl.ckpt" >&2
            echo "[PERF-TEST]        请从 https://huggingface.co/hustvl/DiffusionDriveV2 下载后放入 ckpts/" >&2
            exit 1
        fi ;;
esac

RC=0
run_stage() {
    # $1 = rl|sel; $2 = SEL 阶段初始权重 (仅 sel 需要, 通过环境变量传入 train.sh)
    CHECKPOINT_SEL="${2:-}" bash migrate_to_ascend/train.sh \
        --stage "$1" --rl_epochs "${RL_EPOCHS}" --sel_epochs "${SEL_EPOCHS}" \
        --rank_size "${RANK_SIZE}" --batch_size "${BATCH_SIZE}"
    return $?
}
if [[ "${STAGE}" == "all" || "${STAGE}" == "rl" ]]; then run_stage rl ""; RL_RC=$?; [[ ${RL_RC} -ne 0 ]] && RC=${RL_RC}; fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "sel" ]]; then run_stage sel ckpts/diffusiondrivev2_rl.ckpt; SEL_RC=$?; [[ ${SEL_RC} -ne 0 ]] && RC=${SEL_RC}; fi

# 结果汇总 (解析 train.sh 日志中的 PERF-SUMMARY), 并落盘性能报告 (只含实际跑过的阶段)。
REPORT="${OUT_DIR}/performance_report.txt"
DEVICE_TYPE=$(uname -m)
{
    echo "Network = DiffusionDriveV2"
    echo "DeviceType = ${DEVICE_TYPE}"
    echo "RankSize = ${RANK_SIZE}"
    echo "BatchSize = ${BATCH_SIZE} / device (global $((BATCH_SIZE * RANK_SIZE)))"
    echo "Epochs = RL ${RL_EPOCHS} | SEL ${SEL_EPOCHS}"
    echo "CaseName = DiffusionDriveV2_bs${BATCH_SIZE}_${RANK_SIZE}p_perf"
} > "${REPORT}"

echo ""
echo "------------------ Final result ------------------"
for S in rl sel; do
    case "${STAGE}" in all|"${S}") ;; *) continue ;; esac
    SUM=$(grep -a "PERF-SUMMARY" "${OUT_DIR}/${S}.log" 2>/dev/null | tail -1)
    if [[ -n "${SUM}" ]]; then
        E2E=$(echo "${SUM}"  | grep -oaE 'total_s=[0-9.]+' | head -1 | cut -d= -f2)
        FPS=$(echo "${SUM}"  | grep -oaE 'fps=[0-9.]+'   | head -1 | cut -d= -f2)
        AVG=$(echo "${SUM}"  | grep -oaE 'avg_s=[0-9.]+' | head -1 | cut -d= -f2)
        STEPS=$(echo "${SUM}" | grep -oaE 'n_steps=[0-9]+' | head -1 | cut -d= -f2)
    else
        E2E=""; FPS=""; AVG=""; STEPS=""
    fi
    # 指标统一 "Key = Value" 风格 (与 accuracy_report.txt 一致)。
    echo "${S^^} stage:"
    echo "  FPS = ${FPS:-N/A}"
    echo "  E2ETrainingTime = ${E2E:-N/A}"
    echo "  StepTime = ${AVG:-N/A}"
    echo "  Steps = ${STEPS:-N/A}"
    {
        echo "${S^^} stage:"
        echo "  FPS = ${FPS:-N/A}"
        echo "  E2ETrainingTime = ${E2E:-N/A}"
        echo "  StepTime = ${AVG:-N/A}"
        echo "  Steps = ${STEPS:-N/A}"
    } >> "${REPORT}"
done
echo ""
echo "[PERF-TEST] 性能报告: ${REPORT}"
exit ${RC}
