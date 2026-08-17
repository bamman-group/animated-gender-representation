from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yunet_train.tasks.pose import CocoJsonPoseDataset


def _keypoints_flat_xyv(x: float, y: float, v: float) -> list[float]:
    flat: list[float] = []
    for _ in range(17):
        flat.extend([x, y, v])
    return flat


def test_coco_json_pose_dataset_loads_one_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    assert cv2.imwrite(str(image_dir / "1.jpg"), np.zeros((120, 160, 3), dtype=np.uint8))

    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg", "width": 160, "height": 120}],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [30.0, 20.0, 60.0, 80.0],
                        "iscrowd": 0,
                        "keypoints": _keypoints_flat_xyv(50.0, 50.0, 2.0),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    ds = CocoJsonPoseDataset(ann_path, image_dir)
    assert len(ds) == 1
    sample = ds[0]
    assert sample.boxes.shape == (1, 4)
    assert sample.keypoints.shape == (1, 17, 3)
    assert sample.labels.tolist() == [0]
    np.testing.assert_allclose(sample.keypoints[0, 0], [50.0, 50.0, 2.0])


def test_coco_json_pose_dataset_skips_wrong_keypoint_length(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    assert cv2.imwrite(str(image_dir / "1.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))

    ann_path = tmp_path / "bad.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg", "width": 32, "height": 32}],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0.0, 0.0, 10.0, 10.0],
                        "iscrowd": 0,
                        "keypoints": [0.0] * 30,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No person keypoint"):
        CocoJsonPoseDataset(ann_path, image_dir)
