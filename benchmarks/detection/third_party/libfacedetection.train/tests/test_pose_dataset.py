from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from yunet_train.tasks.pose import COCO8_POSE_ROOT

import pytest
import torch
from torch.utils.data import DataLoader

from yunet_train.tasks.pose import YOLOPoseDataset, build_pose_train_transforms, collate_pose_samples


def _coco8_pose_root() -> Path:
    return COCO8_POSE_ROOT


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_yolo_pose_dataset_loads_coco8_pose_labels() -> None:
    dataset = YOLOPoseDataset(_coco8_pose_root(), split="train")

    assert len(dataset) == 4
    sample = dataset[0]
    height, width = sample.image.shape[:2]
    assert sample.labels.dtype.name == "int64"
    assert sample.boxes.shape[1] == 4
    assert sample.keypoints.shape[1:] == (17, 3)
    assert (sample.boxes[:, 0::2] >= 0).all()
    assert (sample.boxes[:, 0::2] <= width).all()
    assert (sample.boxes[:, 1::2] >= 0).all()
    assert (sample.boxes[:, 1::2] <= height).all()
    assert set(sample.labels.tolist()) <= {0}
    assert set(sample.keypoints[..., 2].reshape(-1).tolist()) <= {0.0, 1.0, 2.0}


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_yolo_pose_dataset_collates_training_batch() -> None:
    dataset = YOLOPoseDataset(_coco8_pose_root(), split="train", transform=build_pose_train_transforms(image_size=128, random_crop=False))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_pose_samples)
    batch = next(iter(loader))

    assert isinstance(batch.images, torch.Tensor)
    assert tuple(batch.images.shape) == (2, 3, 128, 128)
    assert len(batch.boxes) == 2
    assert len(batch.keypoints) == 2
    assert batch.keypoints[0].shape[1:] == (17, 3)
    assert batch.metas[0]["kpt_shape"] == (17, 3)


def test_yolo_pose_dataset_resolves_coco2017_split_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train2017"
    label_dir = tmp_path / "labels" / "train2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "000000000001.jpg"
    label_path = label_dir / "000000000001.txt"

    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    keypoints = " ".join(["0.5 0.5 2"] * 17)
    label_path.write_text(f"0 0.5 0.5 0.5 0.5 {keypoints}\n", encoding="utf-8")

    dataset = YOLOPoseDataset(tmp_path, split="train")

    assert len(dataset) == 1
    assert dataset.image_dir == image_dir
    assert dataset.label_dir == label_dir
    assert dataset[0].keypoints.shape == (1, 17, 3)
