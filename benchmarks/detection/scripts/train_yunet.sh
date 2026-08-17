#!/usr/bin/env bash
# Fine-tune YuNet (libfacedetection.train) from the released WIDER Face
# checkpoint (third_party/libfacedetection.train/weights/yunet_n.pth) on the
# icf and wf_icf settings.
#
# There is no wf run: the released checkpoint IS the widerface-trained model,
# and evaluate_all.sh uses it directly for that condition.
#
#   bash scripts/train_yunet.sh
#   CUDA_VISIBLE_DEVICES=3 SETTINGS=wf_icf bash scripts/train_yunet.sh
source "$(dirname "$0")/train_common.sh"

SETTINGS=${SETTINGS:-"icf wf_icf"}

for s in $SETTINGS; do
    echo ">>> yunet / $s"
    python -m src.train.train_yunet --setting $s --epochs "$EPOCHS" $TEST_FLAG
done
