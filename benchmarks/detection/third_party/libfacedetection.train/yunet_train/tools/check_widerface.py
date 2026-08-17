from __future__ import annotations

import argparse
from pathlib import Path

from yunet_train.tasks.face import (
    WIDER_TRAIN_ANN_FILE,
    WIDER_TRAIN_IMAGE_DIR,
    WIDER_VAL_ANN_FILE,
    WIDER_VAL_IMAGE_DIR,
    WIDERFACE_ROOT,
    parse_labelv2_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check WIDER Face paths before training.")
    parser.add_argument("--root", type=Path, default=WIDERFACE_ROOT)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--check-images", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_file, image_dir = _resolve_split_paths(args.root, args.split)
    test_mode = args.split != "train"

    print(f"root: {args.root}")
    print(f"annotations: {ann_file} ({'ok' if ann_file.exists() else 'missing'})")
    print(f"images: {image_dir} ({'ok' if image_dir.exists() else 'missing'})")
    if not ann_file.exists():
        raise SystemExit(1)

    records = parse_labelv2_file(ann_file, test_mode=test_mode)
    print(f"records: {len(records)}")
    print(f"first record: {records[0].filename if records else 'none'}")

    missing = []
    for record in records[: args.check_images]:
        image_path = image_dir / record.filename
        if not image_path.exists():
            missing.append(image_path)

    if missing:
        print(f"missing images in first {args.check_images}: {len(missing)}")
        for image_path in missing[:5]:
            print(f"  missing: {image_path}")
        raise SystemExit(2)

    print(f"checked images: {min(args.check_images, len(records))}")


def _resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    if root == WIDERFACE_ROOT:
        if split == "train":
            return WIDER_TRAIN_ANN_FILE, WIDER_TRAIN_IMAGE_DIR
        return WIDER_VAL_ANN_FILE, WIDER_VAL_IMAGE_DIR

    ann_file = root / "labelv2.txt"
    image_dir = root / "images"
    if ann_file.exists() or image_dir.exists():
        return ann_file, image_dir

    return root / "labelv2" / split / "labelv2.txt", root / ("WIDER_train" if split == "train" else "WIDER_val") / "images"


if __name__ == "__main__":
    main()
