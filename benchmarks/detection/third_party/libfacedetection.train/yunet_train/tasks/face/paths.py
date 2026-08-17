from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
WIDERFACE_ROOT = DATA_ROOT / "widerface"
WIDER_TRAIN_ROOT = WIDERFACE_ROOT / "WIDER_train"
WIDER_VAL_ROOT = WIDERFACE_ROOT / "WIDER_val"
WIDER_TRAIN_ANN_FILE = WIDERFACE_ROOT / "labelv2" / "train" / "labelv2.txt"
WIDER_VAL_ANN_FILE = WIDERFACE_ROOT / "labelv2" / "val" / "labelv2.txt"
WIDER_TRAIN_IMAGE_DIR = WIDER_TRAIN_ROOT / "images"
WIDER_VAL_IMAGE_DIR = WIDER_VAL_ROOT / "images"
WIDER_VAL_GT_DIR = WIDERFACE_ROOT / "labelv2" / "val" / "gt"
