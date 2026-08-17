from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
from torch.utils.data import Dataset

from .types import FaceRecord, FaceSample
from .widerface import parse_labelv2_file


class WIDERFaceDataset(Dataset):
    def __init__(
        self,
        ann_file: str | Path,
        img_prefix: str | Path,
        *,
        transform: Callable[[FaceSample], FaceSample] | None = None,
        test_mode: bool = False,
        min_size: int | None = None,
    ):
        self.ann_file = Path(ann_file)
        self.img_prefix = Path(img_prefix)
        self.transform = transform
        self.test_mode = test_mode
        self.records = parse_labelv2_file(
            self.ann_file,
            min_size=min_size,
            test_mode=test_mode,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FaceSample:
        record = self.records[index]
        sample = self._record_to_sample(record)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _record_to_sample(self, record: FaceRecord) -> FaceSample:
        image_path = self.img_prefix / record.filename
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        return FaceSample(
            image=image,
            boxes=record.boxes.copy(),
            labels=record.labels.copy(),
            keypoints=record.keypoints.copy(),
            ignored_boxes=record.ignored_boxes.copy(),
            ignored_labels=record.ignored_labels.copy(),
            filename=record.filename,
            original_shape=image.shape,
            image_shape=image.shape,
            pad_shape=image.shape,
        )

