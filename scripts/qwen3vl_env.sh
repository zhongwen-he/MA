#!/usr/bin/env bash

source /home/dellpro2/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl
export PYTHONNOUSERSITE=1
export QWEN3VL_REPO=/home/dellpro2/zhongwen/nuscenes-devkit/third_party/Qwen3-VL
export FORCE_QWENVL_VIDEO_READER=torchvision
