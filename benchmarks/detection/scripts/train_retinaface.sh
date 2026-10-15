#!/usr/bin/env bash
# Fine-tune RetinaFace from the released WIDER Face checkpoint
# (weights/Resnet50_Final.pth) on the icf and wf_icf settings.
#
# There is no wf run: the released checkpoint IS the widerface-trained model,
# and evaluate_all.sh uses it directly for that condition.
#
#   bash scripts/train_retinaface.sh
#   CUDA_VISIBLE_DEVICES=0 SETTINGS=icf bash scripts/train_retinaface.sh
source "$(dirname "$0")/train_common.sh"

SETTINGS=${SETTINGS:-"icf wf_icf"}

for s in $SETTINGS; do
    echo ">>> retinaface / $s"
    python -m src.train.train_retinaface --setting $s \
        --epochs "$EPOCHS" --batch-size "$BATCH" --amp $TEST_FLAG
done
