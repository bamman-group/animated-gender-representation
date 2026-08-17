#!/usr/bin/env bash
# Benchmarks inference throughput (src/benchmark_inference.py) for all 18
# fine-tuned checkpoints across the three training variants (identity,
# tracks, trainonly) x two crop modes (0%/25%) x three architectures
# (buffalo_l, dino_vitb14, dino_vitl14), on a single GPU, appending every
# row to one shared markdown report.
#
# src/benchmark_inference.py only takes one checkpoint per architecture per
# invocation, so this runs it six times (one per variant x crop combination,
# each covering all three archs together) against the same --output file.
# Baselines don't depend on any checkpoint, so they're only measured once
# (the first invocation) - every later invocation passes --skip-baselines.
# --variant-label follows this repo's "<model> (<variant>, crop N%)" --tag
# convention (src/collect_results.py) so the six invocations' rows stay
# distinguishable in the shared output.
#
# Usage:
#   bash scripts/benchmark_all.sh
#   GPU=0 OUTPUT=results/inference_timing.md bash scripts/benchmark_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT=${OUTPUT:-results/inference_timing.md}
DINOV2_DIR=${DINOV2_DIR:-third_party/dinov2}
LOG_DIR=${LOG_DIR:-logs}

IDENTITY_ROOT=outputs/identity
TRACKS_ROOT=outputs/tracks
TRAINONLY_ROOT=outputs/trainonly

mkdir -p "$LOG_DIR"

# Each run's fixed set of common flags; only --variant-label / the three
# --*-checkpoint paths / --crop-padding / --skip-baselines change per call.
run_benchmark () {
    local variant_label=$1
    local crop=$2
    local buffalo_ckpt=$3
    local vitb14_ckpt=$4
    local vitl14_ckpt=$5
    local skip_baselines_flag=$6
    local padding
    padding=$(awk "BEGIN{print ${crop}/100}")
    local log="${LOG_DIR}/benchmark_$(echo "$variant_label" | tr '/ %' '_')_crop${crop}.log"

    echo ">>> Benchmarking variant='${variant_label}' crop=${crop}% -> $log"
    python -m src.benchmark_inference \
        --gpu "$GPU" --batch-size "$BATCH_SIZE" --crop-padding "$padding" \
        --variant-label "${variant_label}, crop ${crop}%" \
        --buffalo-checkpoint "$buffalo_ckpt" \
        --dinov2-dir "$DINOV2_DIR" \
        --dino-vitb14-checkpoint "$vitb14_ckpt" \
        --dino-vitl14-checkpoint "$vitl14_ckpt" \
        $skip_baselines_flag \
        --output "$OUTPUT" 2>&1 | tee "$log"
}

# 1) identity - crop 0%, baselines measured here (first invocation only)
run_benchmark "fine-tuned" 0 \
    "${IDENTITY_ROOT}/buffalo_identity_crop0/backbone_best.pth" \
    "${IDENTITY_ROOT}/dino_vitb14_identity_crop0/backbone_best.pth" \
    "${IDENTITY_ROOT}/dino_vitl14_identity_crop0/backbone_best.pth" \
    ""

# 2) identity - crop 25%
run_benchmark "fine-tuned" 25 \
    "${IDENTITY_ROOT}/buffalo_identity_crop25/backbone_best.pth" \
    "${IDENTITY_ROOT}/dino_vitb14_identity_crop25/backbone_best.pth" \
    "${IDENTITY_ROOT}/dino_vitl14_identity_crop25/backbone_best.pth" \
    "--skip-baselines"

# 3) identity+tracks - crop 0%
run_benchmark "identity+tracks" 0 \
    "${TRACKS_ROOT}/buffalo_tracks_crop0/backbone_best.pth" \
    "${TRACKS_ROOT}/dino_vitb14_tracks_crop0/backbone_best.pth" \
    "${TRACKS_ROOT}/dino_vitl14_tracks_crop0/backbone_best.pth" \
    "--skip-baselines"

# 4) identity+tracks - crop 25%
run_benchmark "identity+tracks" 25 \
    "${TRACKS_ROOT}/buffalo_tracks_crop25/backbone_best.pth" \
    "${TRACKS_ROOT}/dino_vitb14_tracks_crop25/backbone_best.pth" \
    "${TRACKS_ROOT}/dino_vitl14_tracks_crop25/backbone_best.pth" \
    "--skip-baselines"

# 5) identity+tracks/train - crop 0%
run_benchmark "identity+tracks/train" 0 \
    "${TRAINONLY_ROOT}/buffalo_trainonly_tracks_crop0/backbone_best.pth" \
    "${TRAINONLY_ROOT}/dino_vitb14_trainonly_tracks_crop0/backbone_best.pth" \
    "${TRAINONLY_ROOT}/dino_vitl14_trainonly_tracks_crop0/backbone_best.pth" \
    "--skip-baselines"

# 6) identity+tracks/train - crop 25%
run_benchmark "identity+tracks/train" 25 \
    "${TRAINONLY_ROOT}/buffalo_trainonly_tracks_crop25/backbone_best.pth" \
    "${TRAINONLY_ROOT}/dino_vitb14_trainonly_tracks_crop25/backbone_best.pth" \
    "${TRAINONLY_ROOT}/dino_vitl14_trainonly_tracks_crop25/backbone_best.pth" \
    "--skip-baselines"

echo
echo "==================== RESULTS ===================="
[ -f "$OUTPUT" ] && cat "$OUTPUT"
