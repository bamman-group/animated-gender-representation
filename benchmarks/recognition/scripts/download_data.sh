#!/usr/bin/env bash
# Downloads the iCartoonFace dataset and the official recognition evaluation
# code (which contains the complete icartoonface_rectest_info.txt, including
# the probe/gallery pair list) into data/raw.
#
# Usage:
#   bash scripts/download_data.sh
#
# Requires: gdown (installed via environment.yml / pip), and the conda env activated.
set -euo pipefail

FOLDER_ID="1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW"
EVAL_CODE_FILE_ID="1G3g1PslSleSDIEVWqtDIxFLLCNHLtJws"
DEST_DIR="data/raw"

mkdir -p "${DEST_DIR}"

echo "Downloading iCartoonFace data from Google Drive folder ${FOLDER_ID} into ${DEST_DIR} ..."
echo "If this fails (Drive folder downloads are frequently rate-limited or blocked for large"
echo "folders), download manually instead:"
echo "  1. Open https://drive.google.com/drive/folders/${FOLDER_ID}"
echo "  2. Select all files -> right click -> Download (Drive will zip them)"
echo "  3. Unzip the result into ${DEST_DIR}"
echo

gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${DEST_DIR}"

echo
echo "Downloading the official recognition evaluation code (icartoonface_rectest_info.txt"
echo "with the full probe/gallery pair list lives here, not in the Drive folder above) ..."
gdown "https://drive.google.com/uc?id=${EVAL_CODE_FILE_ID}" -O "${DEST_DIR}/icartoonface_rec_evaluation_code.zip"

echo "Download complete. Contents of ${DEST_DIR}:"
ls -la "${DEST_DIR}"

echo
echo "Next step: run 'python scripts/prepare_data.py' to unpack archives and lay out"
echo "the default paths the eval/train scripts expect."
