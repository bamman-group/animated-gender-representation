#!/usr/bin/env bash
# One-time setup: pretrained weights, label files, and training datasets.
# Run from the repo root AFTER placing the datasets under data/
# (see data/README.md):
#   conda env create -f environment.yml && conda activate animated-face
#   bash scripts/setup.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# note: both third_party/ components are vendored (version-controlled), so
# there is nothing to clone here. The libfacedetection.train package is used
# directly from its vendored copy (train_yunet.py runs inside it;
# evaluate_yunet.py adds it to sys.path) - no pip install needed, and its
# pyproject.toml is in fact not installable with setuptools >= 77

echo "== pretrained weights =="
mkdir -p weights results labels
if [ ! -f weights/Resnet50_Final.pth ]; then
    # official folder from the biubug6/Pytorch_Retinaface README
    gdown --folder 1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1 -O weights_dl \
        && mv weights_dl/*.pth weights/ && rm -rf weights_dl
    [ -f weights/Resnet50_Final.pth ] \
        || echo "WARNING: download Resnet50_Final.pth manually (see README) into weights/"
fi

echo "== label files =="
if [ ! -s labels/icf_train.txt ]; then
    python -m src.prepare.prepare_icartoonface \
        --csv data/icartoonface/dettrain.csv \
        --images-dir data/icartoonface/dettrain \
        --output labels/icf_train.txt
fi
if [ ! -s labels/wf_train.txt ]; then
    python -m src.prepare.prepare_widerface \
        --bbx-gt data/widerface/wider_face_split/wider_face_train_bbx_gt.txt \
        --images-dir data/widerface/WIDER_train/images \
        --output labels/wf_train.txt
fi
if [ ! -s labels/icf_train45.txt ] || [ ! -s labels/icf_val.csv ]; then
    python -m src.prepare.split_icartoon_trainval \
        --label-file labels/icf_train.txt --val-n 5000 \
        --out-train labels/icf_train45.txt --out-val labels/icf_val.csv
fi
if [ ! -s labels/film_val.csv ]; then
    python -m src.prepare.prepare_film \
        --annotations data/film/annotations.json \
        --images-root data/film/images \
        --output labels/film_val.csv
fi

echo "== training datasets (symlink layouts) =="
[ -s datasets/wf_icf/data.yaml ] \
    || python -m src.prepare.prepare_yolo_dataset --settings wf icf wf_icf
[ -s datasets/yunet_wf_icf/labelv2.txt ] \
    || python -m src.prepare.prepare_yunet_dataset --settings icf wf_icf

echo "== verify =="
MISSING=0
for f in weights/Resnet50_Final.pth \
         labels/icf_train.txt labels/wf_train.txt labels/film_val.csv \
         labels/icf_train45.txt labels/icf_val.csv \
         datasets/wf/data.yaml datasets/icf/data.yaml datasets/wf_icf/data.yaml \
         datasets/yunet_icf/labelv2.txt datasets/yunet_wf_icf/labelv2.txt; do
    [ -s "$f" ] || { echo "MISSING: $f"; MISSING=1; }
done
if [ "$MISSING" = 0 ]; then
    echo "== done: all setup artifacts present =="
else
    echo "== done WITH MISSING ARTIFACTS - fix the inputs above and rerun =="
    exit 1
fi
