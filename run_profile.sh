#!/bin/bash

GPU=2
GET_FLOPS=0
INPUT_SIZE="1152 1792"      # HD. (w, h)
#INPUT_SIZE="2304 3456"      # 4K. (w, h)
SRC_TRAIN_DIR=./train_root/model_name

python test_profile.py \
    --model_version model \
    --gpus $GPU \
    --input_size $INPUT_SIZE \
    --get_flops $GET_FLOPS \
    --src_train_dir $SRC_TRAIN_DIR