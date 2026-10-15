#!/usr/bin/env bash
# Fine-tune both backbones on the iCartoonFace rectrain identities
# (--data-source identity --loss arcface, the default) on a local GPU server,
# appending every dev/eval row to one shared results.md / timing.jsonl.
# Checkpoints land in $OUTPUT_ROOT/<model>_identity_crop${CROP_PCT}/
# backbone_best.pth, which scripts/evaluate_all.sh picks up automatically.
#
# CROP_PCT selects the face-crop mode (percent of the bbox added per side;
# same convention as the SLURM tracks jobs): 0 = tight crop (default), 25 =
# 25% context padding. Training crops, dev-eval crops, and the checkpoint's
# recorded padding all use it, and output dirs / result tags are suffixed
# with it, so both modes can be trained side by side:
#   CROP_PCT=0  bash scripts/train_all.sh
#   CROP_PCT=25 bash scripts/train_all.sh
#
# TEST=1 runs a minutes-long end-to-end smoke test (1 epoch, a few identities)
# instead of a full run. Override any of the env vars below to change paths /
# sizes, e.g.  ARCHS=vitl14 EPOCHS=40 GPU=1 bash scripts/train_all.sh
#
# ARCHS is the space-separated list of DINOv2 backbones to fine-tune (default
# "vitb14 vitl14"); buffalo_l is trained once regardless, since it has no arch
# variants. scripts/setup.sh only downloads vitb14 weights, so fetch the rest
# first:  bash scripts/download_dino.sh vitl14
#
# (The --data-source tracks runs go to SLURM instead - see
# scripts/slurm/finetune_{buffalo,dino}_tracks.job and the README's
# "Face-track training" section.)
set -uo pipefail
cd "$(dirname "$0")/.."

RESULTS=${RESULTS:-results.md}
TIMING=${TIMING:-timing.jsonl}
ARCHS=${ARCHS:-"vitb14 vitl14"}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
CROP_PCT=${CROP_PCT:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
DINOV2_DIR=${DINOV2_DIR:-third_party/dinov2}
WEIGHTS_DIR=${WEIGHTS_DIR:-data/dinov2_weights}
ONNX=${ONNX:-$HOME/.insightface/models/buffalo_l/w600k_r50.onnx}

PADDING=$(awk "BEGIN{print ${CROP_PCT}/100}")
TEST_FLAG=${TEST:+--test}
FP16_FLAG=""; [ "${FP16:-1}" = "1" ] && FP16_FLAG="--fp16"

echo ">>> Fine-tune InsightFace buffalo_l (identity, ArcFace, crop ${CROP_PCT}%)"
python -m src.train_buffalo \
    --onnx "$ONNX" \
    --crop-padding "$PADDING" \
    --tag "buffalo_l (fine-tuned, crop ${CROP_PCT}%)" \
    --output-dir "$OUTPUT_ROOT/buffalo_identity_crop${CROP_PCT}" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --gpu "$GPU" $FP16_FLAG \
    --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG

for arch in $ARCHS; do
    weights="$WEIGHTS_DIR/dinov2_${arch}_pretrain.pth"
    if [ ! -f "$weights" ]; then
        echo "!! SKIP DINOv2 $arch: no pretrained weights at $weights"
        echo "   (bash scripts/download_dino.sh $arch)"
        continue
    fi
    echo ">>> Fine-tune DINOv2 $arch (identity, ArcFace, crop ${CROP_PCT}%)"
    python -m src.train_dino \
        --dinov2-dir "$DINOV2_DIR" --weights "$weights" --arch "$arch" --proj-dim 0 \
        --crop-padding "$PADDING" \
        --tag "dino_${arch} (fine-tuned, crop ${CROP_PCT}%)" \
        --output-dir "$OUTPUT_ROOT/dino_${arch}_identity_crop${CROP_PCT}" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --gpu "$GPU" $FP16_FLAG \
        --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
done

echo
echo "Done. Evaluate the fine-tuned checkpoints with:"
echo "  CROP_PCT=${CROP_PCT} bash scripts/evaluate_all.sh"
