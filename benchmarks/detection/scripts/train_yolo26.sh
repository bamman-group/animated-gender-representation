#!/usr/bin/env bash
# Train YOLO26-L from COCO-pretrained weights on all three data settings.
#
#   bash scripts/train_yolo26.sh
#   CUDA_VISIBLE_DEVICES=1 SETTINGS="icf wf_icf" bash scripts/train_yolo26.sh
source "$(dirname "$0")/train_common.sh"

SETTINGS=${SETTINGS:-"wf icf wf_icf"}

for s in $SETTINGS; do
    echo ">>> yolo26 / $s"
    python -m src.train.train_ultralytics --arch yolo26 --setting $s \
        --epochs "$EPOCHS" --batch "$BATCH" $TEST_FLAG
done
