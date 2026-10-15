#!/usr/bin/env bash
# Evaluate every model this repo compares - out-of-the-box baselines always,
# and the fine-tuned checkpoints when they exist - on both test sets
# (iCartoonFace rectest + data/film), appending Rank@1 (with a 95% bootstrap
# CI and im/s) to one shared results.md / timing.jsonl. Missing fine-tuned
# checkpoints are skipped with a note, so this is safe to run before or after
# training.
#
# Covers all four training variants, each on both crop modes: baseline,
# fine-tuned (identity), identity+tracks (full paired-face-tracks), and
# identity+tracks/train (tracks restricted to movies with no character overlap
# with data/film - the contamination-controlled number for the film column).
# The last two need track checkpoints copied back from the SLURM cluster (see
# scripts/slurm/); they're skipped with a note when absent.
#
# CROP_PCT selects the face-crop mode (must match the train_all.sh run whose
# checkpoints you want: 0 = tight crop, default; 25 = 25% context padding).
# It picks the crop-suffixed checkpoint dirs, passes the matching
# --crop-padding to every eval (baselines included), and suffixes the result
# tags, so both modes can be evaluated side by side:
#   CROP_PCT=0  bash scripts/evaluate_all.sh
#   CROP_PCT=25 bash scripts/evaluate_all.sh
#
# TEST=1 evaluates a tiny slice per model (seconds, not a valid number) to
# verify the pipeline end to end. Override env vars as for train_all.sh, e.g.
#   ARCHS=vitl14 GPU=1 bash scripts/evaluate_all.sh
#
# ARCHS is the space-separated list of DINOv2 backbones to evaluate (default
# "vitb14 vitl14", matching train_all.sh); buffalo_l is evaluated once.
set -uo pipefail
cd "$(dirname "$0")/.."

RESULTS=${RESULTS:-results.md}
TIMING=${TIMING:-timing.jsonl}
ARCHS=${ARCHS:-"vitb14 vitl14"}
GPU=${GPU:-0}
CROP_PCT=${CROP_PCT:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
DINOV2_DIR=${DINOV2_DIR:-third_party/dinov2}
WEIGHTS_DIR=${WEIGHTS_DIR:-data/dinov2_weights}
ONNX=${ONNX:-$HOME/.insightface/models/buffalo_l/w600k_r50.onnx}

PADDING=$(awk "BEGIN{print ${CROP_PCT}/100}")
TEST_FLAG=${TEST:+--test}

echo ">>> buffalo_l (out of the box, crop ${CROP_PCT}%)"
python -m src.evaluate_baseline --dataset both --gpu "$GPU" \
    --crop-padding "$PADDING" --tag "buffalo_l (baseline, crop ${CROP_PCT}%)" \
    --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG

for arch in $ARCHS; do
    weights="$WEIGHTS_DIR/dinov2_${arch}_pretrain.pth"
    if [ ! -f "$weights" ]; then
        echo "SKIP DINOv2 $arch (no pretrained weights at $weights -"
        echo "     bash scripts/download_dino.sh $arch)"
        continue
    fi
    echo ">>> DINOv2 $arch (out of the box, crop ${CROP_PCT}%)"
    python -m src.train_dino --eval-only --eval-dataset both --gpu "$GPU" \
        --crop-padding "$PADDING" --tag "dino_${arch} (baseline, crop ${CROP_PCT}%)" \
        --dinov2-dir "$DINOV2_DIR" --weights "$weights" --arch "$arch" --proj-dim 0 \
        --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
done

BUF_CKPT="$OUTPUT_ROOT/buffalo_identity_crop${CROP_PCT}/backbone_best.pth"
if [ -f "$BUF_CKPT" ]; then
    echo ">>> buffalo_l (fine-tuned, crop ${CROP_PCT}%: $BUF_CKPT)"
    python -m src.train_buffalo --eval-only --eval-dataset both --gpu "$GPU" \
        --crop-padding "$PADDING" --tag "buffalo_l (fine-tuned, crop ${CROP_PCT}%)" \
        --resume "$BUF_CKPT" --onnx "$ONNX" \
        --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
else
    echo "SKIP fine-tuned buffalo_l (no checkpoint at $BUF_CKPT -"
    echo "     run CROP_PCT=${CROP_PCT} bash scripts/train_all.sh)"
fi

for arch in $ARCHS; do
    weights="$WEIGHTS_DIR/dinov2_${arch}_pretrain.pth"
    DINO_CKPT="$OUTPUT_ROOT/dino_${arch}_identity_crop${CROP_PCT}/backbone_best.pth"
    if [ -f "$DINO_CKPT" ]; then
        echo ">>> DINOv2 $arch (fine-tuned, crop ${CROP_PCT}%: $DINO_CKPT)"
        python -m src.train_dino --eval-only --eval-dataset both --gpu "$GPU" \
            --crop-padding "$PADDING" --tag "dino_${arch} (fine-tuned, crop ${CROP_PCT}%)" \
            --resume "$DINO_CKPT" \
            --dinov2-dir "$DINOV2_DIR" --weights "$weights" --arch "$arch" --proj-dim 0 \
            --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
    else
        echo "SKIP fine-tuned DINOv2 $arch (no checkpoint at $DINO_CKPT -"
        echo "     run CROP_PCT=${CROP_PCT} bash scripts/train_all.sh, and make sure"
        echo "     ARCHS ($ARCHS) / OUTPUT_ROOT match what you trained with)"
    fi
done

# Tracks-fine-tuned checkpoints, if they've been copied back from the SLURM
# cluster into $OUTPUT_ROOT/<model>_tracks_crop${CROP_PCT}/. The tags assume
# the chained workflow (identity checkpoint -> tracks via INIT_WEIGHTS, as
# submitted by scripts/slurm/submit_tracks_jobs.sh); for a from-scratch tracks run,
# evaluate manually with your own --tag.
BUF_TRACKS_CKPT="$OUTPUT_ROOT/buffalo_tracks_crop${CROP_PCT}/backbone_best.pth"
if [ -f "$BUF_TRACKS_CKPT" ]; then
    echo ">>> buffalo_l (identity+tracks, crop ${CROP_PCT}%: $BUF_TRACKS_CKPT)"
    python -m src.train_buffalo --eval-only --eval-dataset both --gpu "$GPU" \
        --crop-padding "$PADDING" --tag "buffalo_l (identity+tracks, crop ${CROP_PCT}%)" \
        --resume "$BUF_TRACKS_CKPT" --onnx "$ONNX" \
        --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
else
    echo "SKIP tracks-fine-tuned buffalo_l (no checkpoint at $BUF_TRACKS_CKPT -"
    echo "     copy backbone_best.pth back from the SLURM run's output dir)"
fi

for arch in $ARCHS; do
    weights="$WEIGHTS_DIR/dinov2_${arch}_pretrain.pth"
    DINO_TRACKS_CKPT="$OUTPUT_ROOT/dino_${arch}_tracks_crop${CROP_PCT}/backbone_best.pth"
    if [ -f "$DINO_TRACKS_CKPT" ]; then
        echo ">>> DINOv2 $arch (identity+tracks, crop ${CROP_PCT}%: $DINO_TRACKS_CKPT)"
        python -m src.train_dino --eval-only --eval-dataset both --gpu "$GPU" \
            --crop-padding "$PADDING" --tag "dino_${arch} (identity+tracks, crop ${CROP_PCT}%)" \
            --resume "$DINO_TRACKS_CKPT" \
            --dinov2-dir "$DINOV2_DIR" --weights "$weights" --arch "$arch" --proj-dim 0 \
            --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
    else
        echo "SKIP tracks-fine-tuned DINOv2 $arch (no checkpoint at $DINO_TRACKS_CKPT -"
        echo "     copy backbone_best.pth back from the SLURM run's output dir)"
    fi
done

# Train-only tracks checkpoints ($OUTPUT_ROOT/<model>_trainonly_tracks_crop${CROP_PCT}/),
# trained on only the movies with no character overlap with data/film
# (scripts/slurm/submit_trainonly_tracks_jobs.sh). Same closed track data as the
# identity+tracks block above, just a restricted movie subset - so these are
# also copied back from the SLURM cluster, and skipped with a note when absent.
BUF_TRAINONLY_CKPT="$OUTPUT_ROOT/buffalo_trainonly_tracks_crop${CROP_PCT}/backbone_best.pth"
if [ -f "$BUF_TRAINONLY_CKPT" ]; then
    echo ">>> buffalo_l (identity+tracks/train, crop ${CROP_PCT}%: $BUF_TRAINONLY_CKPT)"
    python -m src.train_buffalo --eval-only --eval-dataset both --gpu "$GPU" \
        --crop-padding "$PADDING" --tag "buffalo_l (identity+tracks/train, crop ${CROP_PCT}%)" \
        --resume "$BUF_TRAINONLY_CKPT" --onnx "$ONNX" \
        --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
else
    echo "SKIP trainonly-tracks buffalo_l (no checkpoint at $BUF_TRAINONLY_CKPT -"
    echo "     copy backbone_best.pth back from the SLURM run's output dir)"
fi

for arch in $ARCHS; do
    weights="$WEIGHTS_DIR/dinov2_${arch}_pretrain.pth"
    DINO_TRAINONLY_CKPT="$OUTPUT_ROOT/dino_${arch}_trainonly_tracks_crop${CROP_PCT}/backbone_best.pth"
    if [ -f "$DINO_TRAINONLY_CKPT" ]; then
        echo ">>> DINOv2 $arch (identity+tracks/train, crop ${CROP_PCT}%: $DINO_TRAINONLY_CKPT)"
        python -m src.train_dino --eval-only --eval-dataset both --gpu "$GPU" \
            --crop-padding "$PADDING" --tag "dino_${arch} (identity+tracks/train, crop ${CROP_PCT}%)" \
            --resume "$DINO_TRAINONLY_CKPT" \
            --dinov2-dir "$DINOV2_DIR" --weights "$weights" --arch "$arch" --proj-dim 0 \
            --output "$RESULTS" --timing-output "$TIMING" $TEST_FLAG
    else
        echo "SKIP trainonly-tracks DINOv2 $arch (no checkpoint at $DINO_TRAINONLY_CKPT -"
        echo "     copy backbone_best.pth back from the SLURM run's output dir)"
    fi
done

echo
echo "==================== RESULTS ===================="
[ -f "$RESULTS" ] && cat "$RESULTS"
