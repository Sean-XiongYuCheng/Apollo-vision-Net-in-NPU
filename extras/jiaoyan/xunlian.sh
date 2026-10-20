export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
  8 \
  --work-dir work_dirs/npu_dla_8card_worker8 \
  --no-validate \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    data.workers_per_gpu=8 \
    checkpoint_config.by_epoch=False \
    checkpoint_config.interval=50 \
    checkpoint_config.max_keep_ckpts=2 \
    log_config.interval=10