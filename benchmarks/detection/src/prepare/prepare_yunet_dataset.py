"""Build YuNet (libfacedetection.train) datasets for the icf and wf_icf
training settings: a flat symlinked images/ dir plus a labelv2.txt annotation
file (SCRFD format: '# path width height' headers; rows of corner boxes
followed by five landmark x,y,visibility triplets). No keypoints are used in
this benchmark, so every triplet is written unannotated (-1 -1 -1); the
columns are kept because the upstream label parser requires them.

Writes datasets/yunet_<setting>/{images/, labelv2.txt}.

Example (from the repo root):
    python -m src.prepare.prepare_yunet_dataset --settings icf wf_icf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "icf": [("labels/icf_train45.txt", "data/icartoonface/dettrain")],
    "wf_icf": [("labels/icf_train45.txt", "data/icartoonface/dettrain"),
               ("labels/wf_train.txt", "data/widerface/WIDER_train/images")],
}


def read_label_file(path):
    """Our unified format -> {rel: [(box4, lm10, flag), ...]}"""
    out = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current = []
                out[line[1:].strip()] = current
            else:
                v = [float(x) for x in line.split()]
                current.append((v[:4], v[4:14], int(v[14])))
    return out


def build_setting(setting):
    out = REPO_ROOT / "datasets" / f"yunet_{setting}"

    # Every source must actually contribute - see prepare_yolo_dataset.py.
    # Checked before labelv2.txt is opened, so a failed source leaves no
    # half-built dataset behind for setup.sh's guard to treat as done.
    for label_rel, _ in SOURCES[setting]:
        if not read_label_file(REPO_ROOT / label_rel):
            raise SystemExit(
                f"[{setting}] source {label_rel} is empty - its prepare step "
                f"failed or never ran. Fix it and rerun; do not train {setting} "
                f"without it.")

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_img = n_face = n_bad = 0
    with open(out / "labelv2.txt", "w") as ann:
        for label_rel, images_rel in SOURCES[setting]:
            labels = read_label_file(REPO_ROOT / label_rel)
            root = REPO_ROOT / images_rel
            for rel, faces in labels.items():
                src = root / rel
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except Exception:
                    n_bad += 1
                    continue
                name = rel.replace("/", "__")
                link = img_dir / name
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(os.path.abspath(src))

                ann.write(f"# {name} {w} {h}\n")
                for box, _lm, _flag in faces:
                    # clamp to image bounds: out-of-range coordinates crash
                    # the yunet loss's grid indexing (CUDA device assert)
                    x1 = min(max(box[0], 0), w - 1)
                    y1 = min(max(box[1], 0), h - 1)
                    x2 = min(max(box[2], x1 + 1), w)
                    y2 = min(max(box[3], y1 + 1), h)
                    # no keypoints anywhere in this benchmark: every face is
                    # written unannotated (visibility -1), which zeroes the
                    # criterion's kps weight and so its loss_kps term
                    trip = " ".join("-1 -1 -1" for _ in range(5))
                    ann.write(f"{x1:g} {y1:g} {x2:g} {y2:g} {trip}\n")
                    n_face += 1
                n_img += 1

    print(f"[{setting}] {n_img} images, {n_face} faces "
          f"(all keypoints unannotated), {n_bad} unreadable skipped -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", default=["icf", "wf_icf"],
                    choices=["icf", "wf_icf"])
    args = ap.parse_args()
    for setting in args.settings:
        build_setting(setting)


if __name__ == "__main__":
    main()
