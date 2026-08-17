#!/usr/bin/env bash
# Download the iCartoonFace detection dataset (~4 GB) from the official
# Google Drive folder (https://github.com/luxiangju-PersonAI/iCartoonFace)
# and arrange it under data/icartoonface/ in the layout this repo expects:
#
#   data/icartoonface/dettrain/     50,000 training images
#   data/icartoonface/dettrain.csv  updated-v1.0 boxes
#   data/icartoonface/detval/       10,000 validation images
#   data/icartoonface/detval.csv    validation boxes
#
# Requires: gdown (in environment.yml), unzip, ~8 GB free disk during staging.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/icartoonface
STAGE=data/_icartoonface_download
FOLDER_ID=1ARKrhmGAMwVNr8M9kXgDzMUDhzusLxb7           # detection train/val zips
DETVAL_CSV_ID=1qiHHCP1RvMl6kH017pAV8-QDdcMyy8PR       # detection test labels

mkdir -p "$DEST" "$STAGE"

# ---- fetch ---------------------------------------------------------------
if [ ! -f "$STAGE/personai_icartoonface_dettrain.zip" ] \
   || [ ! -f "$STAGE/personai_icartoonface_detval.zip" ]; then
    echo "== downloading detection zips (~4 GB) =="
    gdown --folder "$FOLDER_ID" -O "$STAGE"
    rm -f "$STAGE"/*.usdz   # unrelated file that ships in the folder
fi

if [ ! -f "$DEST/detval.csv" ]; then
    echo "== downloading validation labels =="
    gdown "$DETVAL_CSV_ID" -O "$DEST/detval.csv"
    head -1 "$DEST/detval.csv" | grep -q "personai_icartoonface_detval" \
        || { echo "ERROR: detval.csv looks wrong"; exit 1; }
fi

# ---- unpack + arrange ------------------------------------------------------
if [ ! -d "$DEST/dettrain" ]; then
    echo "== unpacking training images =="
    unzip -q -o "$STAGE/personai_icartoonface_dettrain.zip" -d "$STAGE"
    mv "$STAGE/personai_icartoonface_dettrain/icartoonface_dettrain" "$DEST/dettrain"
fi

if [ ! -f "$DEST/dettrain.csv" ]; then
    echo "== unpacking updated training annotations =="
    unzip -q -o "$STAGE/personai_icartoonface_dettrain_anno_updatedv1.0.zip" -d "$STAGE"
    find "$STAGE" -name "personai_icartoonface_dettrain_anno_updatedv1.0.csv" \
        -exec mv {} "$DEST/dettrain.csv" \;
    [ -f "$DEST/dettrain.csv" ] \
        || { echo "ERROR: updated annotation csv not found in zip"; exit 1; }
fi

if [ ! -d "$DEST/detval" ]; then
    echo "== unpacking validation images =="
    unzip -q -o "$STAGE/personai_icartoonface_detval.zip" -d "$STAGE"
    if [ -d "$STAGE/personai_icartoonface_detval" ]; then
        mv "$STAGE/personai_icartoonface_detval" "$DEST/detval"
    else  # zip may extract flat
        mkdir -p "$DEST/detval"
        mv "$STAGE"/personai_icartoonface_detval_*.jpg "$DEST/detval/"
    fi
fi

# ---- verify ----------------------------------------------------------------
n_train=$(ls "$DEST/dettrain" | wc -l | tr -d ' ')
n_val=$(ls "$DEST/detval" | wc -l | tr -d ' ')
n_boxes=$(wc -l < "$DEST/dettrain.csv" | tr -d ' ')
echo
echo "train images: $n_train (expect 50000)"
echo "val images:   $n_val (expect 10000)"
echo "train boxes:  $n_boxes (expect 91160)"
if [ "$n_train" = 50000 ] && [ "$n_val" = 10000 ]; then
    echo "OK - you can remove the staging dir: rm -rf $STAGE"
else
    echo "WARNING: counts differ from expected; keeping $STAGE for inspection"
fi
