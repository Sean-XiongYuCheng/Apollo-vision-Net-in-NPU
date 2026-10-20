RUN=work_dirs/npu_dla34_8card_24e_w4
mkdir -p "$RUN"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
set -o pipefail

PORT=29509 bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
  8 \
  --work-dir "$RUN" \
  --seed 0 \
  --cfg-options data.workers_per_gpu=4 \
  2>&1 | tee "$RUN/train.log"