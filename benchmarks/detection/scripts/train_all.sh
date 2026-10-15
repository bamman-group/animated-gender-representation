#!/usr/bin/env bash
# Train every model on every applicable data setting, by running the four
# per-model scripts in sequence. Run from the repo root on a CUDA machine.
#
# This is many GPU-days sequentially. Each per-model script is standalone and
# produces exactly the same runs as the corresponding block here, so the usual
# way to parallelize is one script per GPU:
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_yolo26.sh > train_yolo26.log 2>&1
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train_rtdetr.sh > train_rtdetr.log 2>&1
#   CUDA_VISIBLE_DEVICES=2 bash scripts/train_retinaface.sh > train_retinaface.log 2>&1
#   CUDA_VISIBLE_DEVICES=3 bash scripts/train_yunet.sh > train_yunet.log 2>&1
#   wait
#
# The runs are independent - each writes its own runs/<stack>/<setting>/ - so
# the split changes only wall-clock time, not results.
#
# Settings: wf = WIDER Face, icf = iCartoonFace, wf_icf = both (shuffled mix).
# RetinaFace-wf and YuNet-wf need no training: the released checkpoints
# (weights/Resnet50_Final.pth, third_party/libfacedetection.train/weights/yunet_n.pth)
# ARE the widerface-trained models.
#
# Shared knobs (EPOCHS, BATCH, RTDETR_BATCH, TEST) live in train_common.sh and
# are read by the per-model scripts; setting them here in the environment
# propagates to all four, e.g.  TEST=1 bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")"

bash train_retinaface.sh
bash train_yolo26.sh
bash train_rtdetr.sh
bash train_yunet.sh

echo "All training runs finished. Now: bash scripts/evaluate_all.sh"
