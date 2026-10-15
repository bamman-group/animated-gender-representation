#!/usr/bin/env bash
# Download WIDER Face and arrange it under data/widerface/ in the layout this
# repo expects. Links from http://shuoyang1213.me/WIDERFACE/.
#
# Downloads by default (what training needs):
#   data/widerface/WIDER_train/images/          12,880 training images (1.4 GB)
#   data/widerface/wider_face_split/            official bbox annotations
#
# --all additionally downloads WIDER_val (355 MB) and WIDER_test (1.8 GB),
# which this benchmark does not use.
#
# Requires: gdown, unzip, ~4 GB free disk during staging.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/widerface
STAGE=data/_widerface_download
TRAIN_ID=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M
VAL_ID=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q
TEST_ID=1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T
SPLIT_URL="http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"

ALL=0
[ "${1:-}" = "--all" ] && ALL=1

mkdir -p "$DEST" "$STAGE"

# ---- training images ---------------------------------------------------------
if [ ! -d "$DEST/WIDER_train/images" ]; then
    if [ ! -f "$STAGE/WIDER_train.zip" ]; then
        echo "== downloading WIDER_train.zip (1.4 GB) =="
        gdown "$TRAIN_ID" -O "$STAGE/WIDER_train.zip"
    fi
    echo "== unpacking WIDER_train =="
    unzip -q -o "$STAGE/WIDER_train.zip" -d "$DEST"
fi

# ---- official annotations ------------------------------------------------------
if [ ! -d "$DEST/wider_face_split" ]; then
    echo "== downloading wider_face_split.zip =="
    curl -fsSL -o "$STAGE/wider_face_split.zip" "$SPLIT_URL"
    unzip -q -o "$STAGE/wider_face_split.zip" -d "$DEST"
fi

# ---- optional val/test ----------------------------------------------------------
if [ "$ALL" = 1 ]; then
    for part in val test; do
        id_var=$(echo "${part}_ID" | tr '[:lower:]' '[:upper:]')
        if [ ! -d "$DEST/WIDER_${part}/images" ]; then
            echo "== downloading WIDER_${part}.zip =="
            gdown "${!id_var}" -O "$STAGE/WIDER_${part}.zip"
            unzip -q -o "$STAGE/WIDER_${part}.zip" -d "$DEST"
        fi
    done
fi

# ---- verify ----------------------------------------------------------------------
n_train=$(find "$DEST/WIDER_train/images" -name "*.jpg" | wc -l | tr -d ' ')
echo
echo "train images: $n_train (expect 12880)"
if [ "$n_train" = 12880 ]; then
    echo "OK - you can remove the staging dir: rm -rf $STAGE"
else
    echo "WARNING: counts differ from expected; keeping $STAGE for inspection"
fi
