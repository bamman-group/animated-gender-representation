from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .paths import COCO8_POSE_ROOT
from .types import PoseRecord, PoseSample

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLOPoseDataset(Dataset):
    def __init__(
        self,
        root: str | Path = COCO8_POSE_ROOT,
        *,
        split: str = "train",
        transform: Callable[[PoseSample], PoseSample] | None = None,
        kpt_shape: tuple[int, int] = (17, 3),
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.kpt_shape = kpt_shape
        self.image_dir = self._resolve_split_dir("images", split, image_dir)
        self.label_dir = self._resolve_split_dir("labels", split, label_dir)
        self.records = self._collect_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PoseSample:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {record.image_path}")

        labels, boxes, keypoints = self._read_label_file(record.label_path, image.shape)
        sample = PoseSample(
            image=image,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            filename=record.filename,
            original_shape=image.shape,
            image_shape=image.shape,
            pad_shape=image.shape,
            kpt_shape=self.kpt_shape,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _collect_records(self) -> list[PoseRecord]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Pose image directory does not exist: {self.image_dir}")

        records: list[PoseRecord] = []
        for image_path in sorted(self.image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = image_path.relative_to(self.image_dir)
            label_path = (self.label_dir / rel).with_suffix(".txt")
            records.append(PoseRecord(image_path=image_path, label_path=label_path, filename=rel.as_posix()))
        return records

    def _resolve_split_dir(self, kind: str, split: str, override: str | Path | None) -> Path:
        if override is not None:
            path = Path(override)
            return path if path.is_absolute() else self.root / path

        candidates = [self.root / kind / split]
        if split in {"train", "val"}:
            candidates.append(self.root / kind / f"{split}2017")

        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _read_label_file(
        self,
        label_path: Path,
        image_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kpt_num, kpt_dim = self.kpt_shape
        if not label_path.exists():
            return _empty_targets(kpt_num, kpt_dim)

        rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return _empty_targets(kpt_num, kpt_dim)

        height, width = image_shape[:2]
        labels: list[int] = []
        boxes: list[list[float]] = []
        keypoints: list[np.ndarray] = []
        expected = 5 + kpt_num * kpt_dim
        for line_no, row in enumerate(rows, start=1):
            values = [float(value) for value in row.split()]
            if len(values) != expected:
                raise ValueError(f"{label_path}:{line_no} expected {expected} values, got {len(values)}")

            cls_id = int(values[0])
            cx, cy, box_w, box_h = values[1:5]
            x1 = (cx - box_w / 2.0) * width
            y1 = (cy - box_h / 2.0) * height
            x2 = (cx + box_w / 2.0) * width
            y2 = (cy + box_h / 2.0) * height
            labels.append(cls_id)
            boxes.append([x1, y1, x2, y2])

            kpts = np.array(values[5:], dtype=np.float32).reshape(kpt_num, kpt_dim)
            kpts[:, 0] *= width
            kpts[:, 1] *= height
            keypoints.append(kpts)

        labels_array = np.array(labels, dtype=np.int64)
        boxes_array = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        keypoints_array = np.array(keypoints, dtype=np.float32).reshape(-1, kpt_num, kpt_dim)
        boxes_array = _clip_boxes(boxes_array, width, height)
        keypoints_array = _clip_keypoints(keypoints_array, width, height)
        return labels_array, boxes_array, keypoints_array


def _empty_targets(kpt_num: int, kpt_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, kpt_num, kpt_dim), dtype=np.float32),
    )


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
    return boxes


def _clip_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    if keypoints.size == 0:
        return keypoints
    visible = keypoints[..., 2] > 0 if keypoints.shape[-1] >= 3 else np.ones(keypoints.shape[:2], dtype=bool)
    outside = (keypoints[..., 0] < 0) | (keypoints[..., 0] > width) | (keypoints[..., 1] < 0) | (keypoints[..., 1] > height)
    if keypoints.shape[-1] >= 3:
        keypoints[..., 2] = np.where(visible & ~outside, keypoints[..., 2], 0)
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, width)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, height)
    return keypoints
