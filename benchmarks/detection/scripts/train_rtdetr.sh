#!/usr/bin/env bash
# Train RT-DETR-L from COCO-pretrained weights on all three data settings.
#
# Batch size comes from RTDETR_BATCH (not BATCH) - RT-DETR is the more
# memory-hungry of the two ultralytics archs, so it gets its own knob.
#
#   bash scripts/train_rtdetr.sh
#   CUDA_VISIBLE_DEVICES=2 RTDETR_BATCH=8 bash scripts/train_rtdetr.sh
source "$(dirname "$0")/train_common.sh"

SETTINGS=${SETTINGS:-"wf icf wf_icf"}

for s in $SETTINGS; do
    echo ">>> rtdetr / $s"
    python -m src.train.train_ultralytics --arch rtdetr --setting $s \
        --epochs "$EPOCHS" --batch "$RTDETR_BATCH" $TEST_FLAG
done
