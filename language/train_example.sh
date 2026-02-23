#!/bin/bash
for ov_s in 0 7 15
do
  python train.py \
    --var 0.95 \
    --n-last-layers 1 \
    --over_sampling $ov_s \
    --output_dir runs-main-final \
    --dataset qnli \
    --calib_batches 5 \
    --compress
done
