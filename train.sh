#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python config.py \
    --mode train \
    --gpu 0 \
    --patience 5
