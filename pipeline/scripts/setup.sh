#!/usr/bin/env bash
# One-shot setup for pipeline/recognize_characters.py. Fetches everything the
# runner needs: TransNetV2 shot-detection weights, a DINOv2 checkout, the two
# model checkpoints (RT-DETR detector + fine-tuned DINOv2 embedder), and a
# sample clip with its cast list. Assumes the conda env is already created and
# activated:
#
#   conda env create -f pipeline/environment.yml
#   conda activate animated-characters-pipeline
#   bash pipeline/scripts/setup.sh
#
# Re-running is safe - anything already present is left alone.
set -euo pipefail
cd "$(dirname "$0")/.."

TRANSNET_DIR="${TRANSNET_DIR:-models/transnetv2-weights}"
DINOV2_DIR="${DINOV2_DIR:-third_party/dinov2}"
# Model checkpoints, a sample clip, and its cast list. Override to point at a
# mirror; the per-file paths below are appended to it.
DATA_BASE="${DATA_BASE:-http://yosemite.ischool.berkeley.edu/filmanalytics/animation}"

# --- TransNetV2 weights ---------------------------------------------------
# pip-installing TransNetV2 does not reliably bring these: they are Git LFS
# objects, so a clone without git-lfs yields 132-byte pointer files that fail
# at load time with a "corrupted or missing" IOError. Fetch them straight from
# GitHub's LFS media endpoint, which needs no git-lfs at all.
LFS_BASE="https://media.githubusercontent.com/media/soCzech/TransNetV2/master/inference/transnetv2-weights"

if [ -f "${TRANSNET_DIR}/saved_model.pb" ] && \
   [ "$(wc -c < "${TRANSNET_DIR}/saved_model.pb")" -gt 100000 ]; then
    echo "TransNetV2 weights already present at ${TRANSNET_DIR} - skipping."
else
    echo "Downloading TransNetV2 weights into ${TRANSNET_DIR} ..."
    mkdir -p "${TRANSNET_DIR}/variables"
    for f in saved_model.pb variables/variables.data-00000-of-00001 variables/variables.index; do
        curl -L --fail -o "${TRANSNET_DIR}/${f}" "${LFS_BASE}/${f}"
    done
    # A pointer file is ~130 bytes; the real SavedModel is ~5.9 MB.
    if [ "$(wc -c < "${TRANSNET_DIR}/saved_model.pb")" -lt 100000 ]; then
        echo "!! ${TRANSNET_DIR}/saved_model.pb looks like an LFS pointer, not the model."
        exit 1
    fi
fi

# --- DINOv2 checkout ------------------------------------------------------
# Needed for its hubconf.py: the fine-tuned checkpoint carries the weights but
# not the model definition (see load_dino() in recognize_characters.py).
if [ -d "${DINOV2_DIR}/.git" ]; then
    echo "DINOv2 checkout already exists at ${DINOV2_DIR} - skipping clone."
else
    echo "Cloning https://github.com/facebookresearch/dinov2 into ${DINOV2_DIR} ..."
    mkdir -p "$(dirname "${DINOV2_DIR}")"
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "${DINOV2_DIR}"
fi

# --- Model checkpoints, sample clip, and cast list ------------------------
# Each path mirrors its location on the server and lands at the same relative
# path under pipeline/, which is where recognize_characters.py's flags expect
# it (see the run command below).
for rel in \
    models/rtdetr.icf.FINAL.pt \
    models/dino_vitl14_crop25.FINAL.pth \
    character_embeddings/gullivers.trave_tt0031397.tsv \
    samples/gulliver_clip.mp4 ; do
    if [ -f "$rel" ]; then
        echo "${rel} already present - skipping."
    else
        echo "Downloading ${rel} ..."
        mkdir -p "$(dirname "$rel")"
        curl -L --fail -o "$rel" "${DATA_BASE}/${rel}"
    fi
done

echo
echo "Setup done. Run the sample clip end to end with:"
echo "  python recognize_characters.py samples/gulliver_clip.mp4 \\"
echo "      --transnet ${TRANSNET_DIR} \\"
echo "      --detr models/rtdetr.icf.FINAL.pt \\"
echo "      --dino models/dino_vitl14_crop25.FINAL.pth \\"
echo "      --dinov2-dir ${DINOV2_DIR} \\"
echo "      --cast character_embeddings/gullivers.trave_tt0031397.tsv \\"
echo "      --annotate --out out/"
