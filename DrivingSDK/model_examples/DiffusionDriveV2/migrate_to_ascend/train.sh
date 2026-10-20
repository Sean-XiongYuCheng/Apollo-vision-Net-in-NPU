#!/bin/bash
# DiffusionDriveV2 两阶段训练脚本
#
# 参数:
#   --stage S         运行阶段: all|rl|sel      (默认 all)
#   --rl_epochs N     RL 阶段 epochs            (默认 10)
#   --sel_epochs N    SEL 阶段 epochs           (默认 20)
#   --rank_size N     使用的 NPU 卡数           (默认 8)
#   --batch_size N    每卡 batch size           (默认 64, 官方默认)
#   --help, -h        显示帮助
#
# 环境变量:
#   TRAIN_TEST_SPLIT=navtrain
#   CHECKPOINT_SEL=/path/to/ckpt
#   OUT_DIR=/path/to/log_dir   (默认 migrate_to_ascend/output/full)
#   PDM_WORKERS=N
set -u

# 默认参数。
RL_EPOCHS=${RL_EPOCHS:-10}
SEL_EPOCHS=${SEL_EPOCHS:-20}
RANK_SIZE=${RANK_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
STAGE=${STAGE:-all}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
CHECKPOINT_SEL=${CHECKPOINT_SEL:-}

# 解析命令行参数。
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            cat <<'HELP'
Usage:
  bash train.sh                                              # 8 devices, navtrain, RL 10ep + SEL 20ep
  bash train.sh --stage rl --rl_epochs 2 --rank_size 2        # RL only smoke
  bash train.sh --stage sel --sel_epochs 20 --batch_size 32 --rank_size 8
  TRAIN_TEST_SPLIT=mini bash train.sh --rl_epochs 1 --sel_epochs 1 --rank_size 2   # mini smoke
  CHECKPOINT_SEL=/path/to/ckpt bash train.sh --stage sel ... # explicit SEL init ckpt

Options:
  --stage S          stage to run: all|rl|sel   (default all)
  --rl_epochs N      RL stage epochs            (default 10)
  --sel_epochs N     SEL stage epochs           (default 20)
  --rank_size N      number of NPU devices      (default 8)
  --batch_size N     per-device batch size      (default 64)
  --help, -h         show this help
HELP
            exit 0
            ;;
        --stage)      STAGE=$2; shift 2 ;;
        --rl_epochs)  RL_EPOCHS=$2; shift 2 ;;
        --sel_epochs) SEL_EPOCHS=$2; shift 2 ;;
        --rank_size)  RANK_SIZE=$2; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash train.sh [--stage all|rl|sel] [--rl_epochs N] [--sel_epochs N] [--rank_size N] [--batch_size N]" >&2
            echo "Run 'bash train.sh --help' for details." >&2
            exit 1
            ;;
    esac
done

case "${STAGE}" in all|rl|sel) ;; *)
    echo "[TRAIN] ERROR: --stage 必须是 all|rl|sel (收到: ${STAGE})" >&2; exit 1 ;;
esac

# 日志目录 (可由调用方 export OUT_DIR 指定, 精度/性能各自的入口脚本会指向不同目录)
OUT_DIR=${OUT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output/full"}
mkdir -p "${OUT_DIR}"

# Python 环境
PYTHON=${PYTHON:-python}

# 加载昇腾运行环境。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_npu.sh"

# 训练阶段 (终端只打印一行进度, 明细见各阶段日志)。
run_rl() {
    echo "[TRAIN] RL stage: ${RL_EPOCHS} ep, ${RANK_SIZE} devices, batch ${BATCH_SIZE}/device | 日志: ${OUT_DIR}/rl.log"
    ${PYTHON} ./navsim/planning/script/run_training.py \
        agent=diffusiondrivev2_rl_agent \
        experiment_name=training_diffusiondrivev2_rl_agent \
        train_test_split=${TRAIN_TEST_SPLIT} \
        agent.checkpoint_path=ckpts/diffusiondrive_navsim_88p1_PDMS \
        trainer.params.max_epochs=${RL_EPOCHS} \
        +trainer.params.devices=${RANK_SIZE} \
        dataloader.params.batch_size=${BATCH_SIZE} \
        cache_path="./exp/training_cache/" \
        use_cache_without_dataset=True \
        force_cache_computation=False > "${OUT_DIR}/rl.log" 2>&1
    RL_RC=$?
}

run_sel() {
    # 初始权重: CHECKPOINT_SEL 优先, 否则用 ckpts/ 下的官方 RL 权重。
    local sel_init
    if [ -n "${CHECKPOINT_SEL}" ]; then
        sel_init="${CHECKPOINT_SEL}"
    else
        sel_init="ckpts/diffusiondrivev2_rl.ckpt"
    fi
    # Hydra 参数不接受未转义的 "=", 用软链暴露一个无 "=" 的路径 (软链目标需为绝对路径)。
    if [ -f "${sel_init}" ] && [[ "${sel_init}" == *"="* ]]; then
        [[ "${sel_init}" == /* ]] || sel_init="$(pwd)/${sel_init}"
        ln -sfn "${sel_init}" exp/rl_stage_best.ckpt
        sel_init="exp/rl_stage_best.ckpt"
    fi
    echo "[TRAIN] SEL stage: ${SEL_EPOCHS} ep, ${RANK_SIZE} devices, batch ${BATCH_SIZE}/device | 日志: ${OUT_DIR}/sel.log"
    ${PYTHON} ./navsim/planning/script/run_training.py \
        agent=diffusiondrivev2_sel_agent \
        experiment_name=training_diffusiondrivev2_sel_agent \
        train_test_split=${TRAIN_TEST_SPLIT} \
        agent.checkpoint_path="${sel_init}" \
        trainer.params.max_epochs=${SEL_EPOCHS} \
        +trainer.params.devices=${RANK_SIZE} \
        dataloader.params.batch_size=${BATCH_SIZE} \
        cache_path="./exp/training_cache/" \
        use_cache_without_dataset=True \
        force_cache_computation=False > "${OUT_DIR}/sel.log" 2>&1
    SEL_RC=$?
}

# 按阶段调度。
RL_RC=0; SEL_RC=0
if [[ "${STAGE}" == "all" || "${STAGE}" == "rl" ]]; then run_rl; fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "sel" ]]; then run_sel; fi

# 任一阶段失败则返回非零 (成功时保持安静, "Final result" 由入口脚本统一输出)。
if [[ ${RL_RC} -ne 0 ]]; then echo "[TRAIN] RL stage FAILED (rc=${RL_RC}), 详见 ${OUT_DIR}/rl.log"; fi
if [[ ${SEL_RC} -ne 0 ]]; then echo "[TRAIN] SEL stage FAILED (rc=${SEL_RC}), 详见 ${OUT_DIR}/sel.log"; fi
exit $(( RL_RC != 0 || SEL_RC != 0 ))
