from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from yunet_train.tasks.pose import COCO17_FLIP_IDX, COCO8_POSE_ROOT, YOLOPoseDataset, build_pose_eval_transforms, build_pose_train_transforms
from yunet_train.tasks.pose.visualize import pose_sample_annotation_text, render_pose_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YuNet pose training samples after augmentation.")
    parser.add_argument("--data-root", type=Path, default=COCO8_POSE_ROOT)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/pose_augmentation_debug"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-transform", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_pose_dataset(args)


def visualize_pose_dataset(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    transform = None
    if not args.no_transform:
        transform = (
            build_pose_train_transforms(args.image_size, flip_idx=COCO17_FLIP_IDX)
            if args.split == "train"
            else build_pose_eval_transforms(args.image_size)
        )
    dataset = YOLOPoseDataset(args.data_root, split=args.split, transform=transform, kpt_shape=(17, 3))
    if len(dataset) == 0:
        raise ValueError(f"Dataset has no images: {args.data_root} split={args.split}")

    for epoch in range(1, args.epochs + 1):
        for sample_idx in range(args.samples_per_epoch):
            dataset_idx = int(rng.integers(0, len(dataset)))
            sample = dataset[dataset_idx]
            rendered = render_pose_sample(sample)
            stem = f"epoch_{epoch:03d}_sample_{sample_idx:02d}"
            image_path = args.out_dir / f"{stem}.jpg"
            text_path = args.out_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), rendered)
            text_path.write_text(
                f"dataset_index: {dataset_idx}\n" + pose_sample_annotation_text(sample),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
