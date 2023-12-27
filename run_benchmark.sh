#!/bin/bash

GPU=1
DATASET=P3M     # can be P3M or PPM or PhotoMatte
PRIVACY_DATA=0      # 0 for non privacy data of P3M; 1 for privacy preserving data of P3M; ignored for PPM and PhotoMatte
PROFILE_ONLY=0
GET_FLOPS=0
SAVE_IMGS=0
SRC_TRAIN_DIR=./train_root/model_name

if [ P3M = $DATASET ]; then
    RESIZE_FACTOR=1
    MAX_AREA=None
else
    RESIZE_FACTOR=1
    MAX_AREA=2073600        # HD: 1080*1920
    #MAX_AREA=8294400        # 4K: 2160*3840
fi

python test_benchmark.py \
    --gpus $GPU \
    --model_version model \
    --config "TEST.BENCHMARK.RESIZE_FACTOR=$RESIZE_FACTOR;TEST.BENCHMARK.MAX_AREA=$MAX_AREA" \
    --dataset $DATASET \
    --privacy_data $PRIVACY_DATA \
    --profile_only $PROFILE_ONLY \
    --get_flops $GET_FLOPS \
    --save_imgs $SAVE_IMGS \
    --src_train_dir $SRC_TRAIN_DIR