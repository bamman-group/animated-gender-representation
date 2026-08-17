"""Build Ultralytics datasets (for YOLO26 / RT-DETR) for the three training
settings from the label files produced by prepare_icartoonface.py /
prepare_widerface.py.

Writes, per setting, under datasets/<setting>/:
    images/train/<name>.jpg   (symlinks into data/)
    labels/train/<name>.txt   (class cx cy w h normalized; class 0 = face)
    images/val/, labels/val/  (first --val-limit images of the icartoon_val
                              split, labels/icf_val.csv - held out from
                              dettrain; detval is never used for training-time
                              validation)
    data.yaml

Example (from the repo root):
    python -m src.prepare.prepare_yolo_dataset --settings wf icf wf_icf
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "icf": [("labels/icf_train45.txt", "data/icartoonface/dettrain")],
    "wf": [("labels/wf_train.txt", "data/widerface/WIDER_train/images")],
}
SOURCES["wf_icf"] = SOURCES["icf"] + SOURCES["wf"]


def read_label_file(path):
    boxes = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current = []
                boxes[line[1:].strip()] = current
            else:
                current.append([float(v) for v in line.split()[:4]])
    return boxes


def read_val_csv(path):
    boxes = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            x1, y1, x2, y2 = (float(v) for v in p[1:5])
            if x2 > x1 and y2 > y1:
                boxes[p[0]].append([x1, y1, x2, y2])
    return boxes


def write_split(records, out_dir, split):
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_bad = 0
    for src, name, boxes in records:
        try:
            with Image.open(src) as im:  # header-only read
                w, h = im.size
        except Exception:
            n_bad += 1
            continue
        lines = []
        for x1, y1, x2, y2 in boxes:
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, w), min(y2, h)
            if x2c - x1c < 1 or y2c - y1c < 1:
                continue
            lines.append(f"0 {(x1c + x2c) / 2 / w:.6f} {(y1c + y2c) / 2 / h:.6f} "
                         f"{(x2c - x1c) / w:.6f} {(y2c - y1c) / h:.6f}")
        if not lines:
            continue
        link = img_dir / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.abspath(src))
        (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(lines)
    print(f"  {split}: {n_img} images, {n_box} boxes"
          + (f", {n_bad} unreadable skipped" if n_bad else ""))
    return n_img


def build_setting(setting, val, val_names):
    out = REPO_ROOT / "datasets" / setting
    print(f"[{setting}] -> {out}")

    # Every source must actually contribute. A setting like wf_icf whose wf
    # label file is empty would otherwise build cleanly from icf alone and
    # train as "wf_icf" on icf-only data - a silent mislabel, not a crash.
    for label_rel, _ in SOURCES[setting]:
        if not read_label_file(REPO_ROOT / label_rel):
            raise SystemExit(
                f"[{setting}] source {label_rel} is empty - its prepare step "
                f"failed or never ran. Fix it and rerun; do not train {setting} "
                f"without it.")

    def train_records():
        for label_rel, images_rel in SOURCES[setting]:
            labels = read_label_file(REPO_ROOT / label_rel)
            root = REPO_ROOT / images_rel
            for rel, boxes in labels.items():
                yield str(root / rel), rel.replace("/", "__"), boxes

    n_train = write_split(train_records(), out, "train")
    val_root = REPO_ROOT / "data/icartoonface/dettrain"
    n_val = write_split(((str(val_root / n), n.replace("/", "__"), val[n])
                         for n in val_names), out, "val")

    # Refuse to write data.yaml for an empty split: it would look like a
    # finished dataset to setup.sh's guard/verify, and only fail much later
    # inside the trainer's dataloader.
    if not n_train or not n_val:
        raise SystemExit(
            f"[{setting}] refusing to write data.yaml: "
            f"train={n_train} images, val={n_val} images. "
            f"Check the source label files ({', '.join(s[0] for s in SOURCES[setting])}"
            f", labels/icf_val.csv) - an empty one means its prepare step "
            f"failed or never ran.")

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: face\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", default=["wf", "icf", "wf_icf"],
                    choices=["wf", "icf", "wf_icf"])
    ap.add_argument("--val-limit", type=int, default=2000,
                    help="icartoon_val (train-holdout) images used for "
                         "in-training val (0 = all)")
    args = ap.parse_args()

    val = read_val_csv(REPO_ROOT / "labels/icf_val.csv")
    val_names = sorted(val)
    if args.val_limit > 0:
        val_names = val_names[:args.val_limit]

    for setting in args.settings:
        build_setting(setting, val, val_names)


if __name__ == "__main__":
    main()
