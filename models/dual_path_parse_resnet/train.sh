#!/usr/bin/env sh

LOG='log.log'
nohup tools/caffe train \
    --gpu=7 \
    --solver=solver.prototxt \
    --weights=../../data/pretrained/new_40_attribute_iter_1.caffemodel 2>&1 |tee $LOG


