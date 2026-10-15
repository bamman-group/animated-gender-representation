from __future__ import annotations

from pathlib import Path

import numpy as np

from yunet_train.tasks.face import parse_labelv2_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_train_labelv2_first_records() -> None:
    records = parse_labelv2_file(
        REPO_ROOT / "data" / "widerface" / "labelv2" / "train" / "labelv2.txt"
    )

    assert len(records) > 0
    first = records[0]
    assert first.filename == "0--Parade/0_Parade_marchingband_1_849.jpg"
    assert first.width == 1024
    assert first.height == 1385
    assert first.boxes.shape == (1, 4)
    assert first.keypoints.shape == (1, 5, 3)
    np.testing.assert_allclose(first.boxes[0], [449, 330, 571, 479])
    assert first.labels.tolist() == [0]
    assert first.ignored_boxes.shape == (0, 4)


def test_parse_train_labelv2_ignores_missing_keypoints() -> None:
    records = parse_labelv2_file(
        REPO_ROOT / "data" / "widerface" / "labelv2" / "train" / "labelv2.txt"
    )

    record = records[2]
    assert record.filename == "0--Parade/0_Parade_marchingband_1_799.jpg"
    assert record.boxes.shape[0] == 21
    assert record.keypoints[0, :, 2].tolist() == [0, 0, 0, 0, 0]


def test_parse_val_labelv2_allows_bbox_only_annotations() -> None:
    records = parse_labelv2_file(
        REPO_ROOT / "data" / "widerface" / "labelv2" / "val" / "labelv2.txt",
        test_mode=True,
    )

    assert len(records) > 0
    first = records[0]
    assert first.filename == "0--Parade/0_Parade_marchingband_1_465.jpg"
    assert first.boxes.shape[1] == 4
    assert first.keypoints.shape[1:] == (5, 3)
