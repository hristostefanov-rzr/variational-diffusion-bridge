#!/bin/bash
python sample.py \
--ckpt sr4x-bicubic \
--n-gpu-per-node 1 \
--dataset-dir /home/hristo/Code/thesis/imagenet_data/imagenet-mini \
--batch-size 1 \
--use-fp16 \
--clip-denoise \
--nfe 2 \
--step-size 1.0 \
#--use-cddb-deep
#--dataset-dir /home/hristo/Code/thesis/small_image_net_data/ \