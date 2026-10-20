RUN=work_dirs/bench_dla34_b2
mkdir -p "$RUN"

# 改成另一组空闲卡；不要使用正在正式训练的卡。
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=1
set -o pipefail

PORT=29511 bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
  8 \
  --work-dir "$RUN" \
  --seed 0 \
  --no-validate \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    data.samples_per_gpu=2 \
    data.workers_per_gpu=4 \
    checkpoint_config.interval=100 \
  2>&1 | tee "$RUN/train.log"