#!/usr/bin/env bash
# Downloads the DINOv2 repo checkout + a pretrained backbone checkpoint
# needed by src/train_dino.py's --dinov2-dir/--weights - neither is a pip
# dependency of this project (see src/models/dino_backbone.py).
#
# Usage:
#   bash scripts/download_dino.sh [arch]
#   ARCH=vitb14 bash scripts/download_dino.sh
#
# arch is one of vits14, vitb14 (default), vitl14, vitg14 - matches
# src/models/dino_backbone.py::ARCH_EMBED_DIM and every --arch example in
# README.md. Re-running is safe - an existing checkout/weights file is left
# alone rather than re-downloaded.
set -euo pipefail

ARCH="${1:-${ARCH:-vitb14}}"
case "$ARCH" in
    vits14|vitb14|vitl14|vitg14) ;;
    *)
        echo "Unknown arch: $ARCH (expected one of vits14, vitb14, vitl14, vitg14)"
        exit 1
        ;;
esac

DINOV2_DIR="${DINOV2_DIR:-third_party/dinov2}"
WEIGHTS_DIR="${WEIGHTS_DIR:-data/dinov2_weights}"
WEIGHTS_FILE="${WEIGHTS_DIR}/dinov2_${ARCH}_pretrain.pth"

mkdir -p "$(dirname "${DINOV2_DIR}")" "${WEIGHTS_DIR}"

if [ -d "${DINOV2_DIR}/.git" ]; then
    echo "DINOv2 repo checkout already exists at ${DINOV2_DIR} - skipping clone."
else
    echo "Cloning https://github.com/facebookresearch/dinov2 into ${DINOV2_DIR} ..."
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "${DINOV2_DIR}"
fi

if [ -f "${WEIGHTS_FILE}" ]; then
    echo "Pretrained weights already exist at ${WEIGHTS_FILE} - skipping download."
else
    URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_${ARCH}/dinov2_${ARCH}_pretrain.pth"
    echo "Downloading ${ARCH} pretrained weights from ${URL} ..."
    curl -L --fail -o "${WEIGHTS_FILE}" "${URL}"
fi

echo
echo "Done. Pass these to any DINOv2 command in this repo:"
echo "  --dinov2-dir ${DINOV2_DIR} --weights ${WEIGHTS_FILE} --arch ${ARCH}"
