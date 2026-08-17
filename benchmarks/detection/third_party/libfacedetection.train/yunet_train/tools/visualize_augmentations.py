from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from yunet_train.tasks.face import get_train_crop_choice
from yunet_train.tasks.face import WIDERFaceDataset, WIDER_TRAIN_ANN_FILE, WIDER_TRAIN_IMAGE_DIR, build_train_transforms, collate_face_samples

KEYPOINT_COLORS = (
    (255, 96, 0),
    (0, 192, 255),
    (0, 255, 255),
    (255, 0, 255),
    (160, 64, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize real WIDERFace train-time augmentations.")
    parser.add_argument("--variant", default="yunet_s", choices=("yunet_n", "yunet_s"))
    parser.add_argument("--ann-file", type=Path, default=WIDER_TRAIN_ANN_FILE)
    parser.add_argument("--img-prefix", type=Path, default=WIDER_TRAIN_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/augmentation_debug"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--min-face-size", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = visualize_augmentations(args)
    print(f"saved {len(saved)} visualizations to {args.output_dir}")
    for path in saved:
        print(path)


def visualize_augmentations(args: argparse.Namespace) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = WIDERFaceDataset(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        transform=build_train_transforms(
            image_size=args.image_size,
            crop_choice=get_train_crop_choice(args.variant),
            min_box_size=args.min_face_size,
        ),
    )
    if args.limit_samples is not None:
        dataset.records = dataset.records[: args.limit_samples]

    saved: list[Path] = []
    for epoch in range(1, args.epochs + 1):
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        data_loader = DataLoader(
            dataset,
            batch_size=args.samples_per_epoch,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_face_samples,
            generator=generator,
        )
        batch = next(iter(data_loader))
        for sample_index, image_tensor in enumerate(batch.images):
            image = image_tensor_to_bgr_uint8(image_tensor)
            meta = batch.metas[sample_index]
            title = f"epoch={epoch} sample={sample_index} flip={meta['flip']} {meta['filename']}"
            canvas = draw_face_annotations(
                image=image,
                boxes=batch.boxes[sample_index],
                keypoints=batch.keypoints[sample_index],
                ignored_boxes=batch.ignored_boxes[sample_index],
                title=title,
            )
            stem = f"epoch_{epoch:03d}_sample_{sample_index:02d}"
            image_path = args.output_dir / f"{stem}.jpg"
            annotation_path = args.output_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), canvas)
            write_annotation_file(
                annotation_path,
                boxes=batch.boxes[sample_index],
                keypoints=batch.keypoints[sample_index],
                ignored_boxes=batch.ignored_boxes[sample_index],
                meta=meta,
            )
            saved.append(image_path)
    return saved


def image_tensor_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image tensor with shape (3, H, W), got {tuple(image.shape)}")
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(array, 0, 255).astype(np.uint8)


def draw_face_annotations(
    *,
    image: np.ndarray,
    boxes: torch.Tensor | np.ndarray,
    keypoints: torch.Tensor | np.ndarray,
    ignored_boxes: torch.Tensor | np.ndarray | None = None,
    title: str | None = None,
) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    _draw_boxes(canvas, _to_numpy(boxes), color=(0, 255, 0), label="face")
    if ignored_boxes is not None:
        _draw_boxes(canvas, _to_numpy(ignored_boxes), color=(0, 0, 255), label="ignore")
    _draw_keypoints(canvas, _to_numpy(keypoints))
    if title:
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def write_annotation_file(
    path: Path,
    *,
    boxes: torch.Tensor | np.ndarray,
    keypoints: torch.Tensor | np.ndarray,
    ignored_boxes: torch.Tensor | np.ndarray,
    meta: dict[str, object],
) -> None:
    lines = [
        f"filename: {meta['filename']}",
        f"flip: {meta['flip']}",
        f"img_shape: {meta['img_shape']}",
        "boxes:",
    ]
    for box in _to_numpy(boxes):
        lines.append("  " + " ".join(f"{value:.2f}" for value in box.tolist()))
    lines.append("ignored_boxes:")
    for box in _to_numpy(ignored_boxes):
        lines.append("  " + " ".join(f"{value:.2f}" for value in box.tolist()))
    lines.append("keypoints:")
    for face_kps in _to_numpy(keypoints):
        flat = face_kps.reshape(-1)
        lines.append("  " + " ".join(f"{value:.2f}" for value in flat.tolist()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_boxes(canvas: np.ndarray, boxes: np.ndarray, *, color: tuple[int, int, int], label: str) -> None:
    for box in boxes.reshape(-1, 4):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_keypoints(canvas: np.ndarray, keypoints: np.ndarray) -> None:
    for face_kps in keypoints.reshape(-1, 5, 3):
        for index, (x, y, visible) in enumerate(face_kps):
            if visible <= 0:
                continue
            cv2.circle(canvas, (int(round(x)), int(round(y))), 3, KEYPOINT_COLORS[index], -1, cv2.LINE_AA)


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


if __name__ == "__main__":
    main()
