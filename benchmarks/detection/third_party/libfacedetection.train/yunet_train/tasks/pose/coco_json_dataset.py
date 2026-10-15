from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from .dataset import _clip_boxes, _clip_keypoints
from .types import PoseSample


@dataclass(frozen=True)
class _CocoImageRecord:
    image_info: dict[str, Any]
    annotations: list[dict[str, Any]]


class CocoJsonPoseDataset(Dataset):
    """COCO human keypoint JSON (e.g. person_keypoints_*.json) + image directory."""

    def __init__(
        self,
        ann_file: str | Path,
        image_dir: str | Path,
        *,
        transform: Callable[[PoseSample], PoseSample] | None = None,
        kpt_shape: tuple[int, int] = (17, 3),
        category_id: int = 1,
    ):
        self.ann_file = Path(ann_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.kpt_shape = kpt_shape
        self.category_id = category_id
        self.records: list[_CocoImageRecord] = self._build_records()
        if not self.records:
            raise ValueError(f"No person keypoint annotations found in {self.ann_file}")

    def _build_records(self) -> list[_CocoImageRecord]:
        data = json.loads(self.ann_file.read_text(encoding="utf-8"))
        images = {int(img["id"]): img for img in data.get("images", [])}
        per_anns: dict[int, list[dict[str, Any]]] = {i: [] for i in images}

        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            if int(ann.get("category_id", -1)) != self.category_id:
                continue
            kpts = ann.get("keypoints")
            if not kpts or len(kpts) != self.kpt_shape[0] * self.kpt_shape[1]:
                continue
            img_id = int(ann["image_id"])
            if img_id not in per_anns:
                continue
            per_anns[img_id].append(ann)

        records: list[_CocoImageRecord] = []
        for img_id in sorted(images.keys()):
            anns = per_anns[img_id]
            if not anns:
                continue
            records.append(_CocoImageRecord(image_info=images[img_id], annotations=anns))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PoseSample:
        rec = self.records[index]
        info = rec.image_info
        file_name = info["file_name"]
        path = self.image_dir / file_name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read COCO image: {path}")

        height, width = image.shape[:2]
        labels: list[int] = []
        boxes: list[list[float]] = []
        keypoints: list[np.ndarray] = []

        for ann in rec.annotations:
            bbox = ann["bbox"]
            x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            raw = np.array(ann["keypoints"], dtype=np.float32).reshape(self.kpt_shape[0], self.kpt_shape[1])
            kpt = raw.copy()

            labels.append(0)
            boxes.append([x1, y1, x2, y2])
            keypoints.append(kpt)

        labels_array = np.array(labels, dtype=np.int64)
        boxes_array = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        keypoints_array = np.array(keypoints, dtype=np.float32).reshape(-1, *self.kpt_shape)
        boxes_array = _clip_boxes(boxes_array, width, height)
        keypoints_array = _clip_keypoints(keypoints_array, width, height)

        sample = PoseSample(
            image=image,
            boxes=boxes_array,
            labels=labels_array,
            keypoints=keypoints_array,
            filename=str(file_name),
            original_shape=image.shape,
            image_shape=image.shape,
            pad_shape=image.shape,
            kpt_shape=self.kpt_shape,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
