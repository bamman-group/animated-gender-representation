from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from yunet_train.tasks.face import (
    FaceSample,
    FilterSmallBoxes,
    Pad,
    RandomHorizontalFlip,
    RandomSquareCrop,
    Resize,
    ToTensor,
    WIDERFaceDataset,
    collate_face_samples,
)


def _sample() -> FaceSample:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(6, dtype=np.uint8)
    return FaceSample(
        image=image,
        boxes=np.array([[1, 1, 4, 3]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        keypoints=np.array(
            [
                [
                    [1, 1, 1],
                    [4, 1, 1],
                    [2, 2, 1],
                    [1, 3, 1],
                    [4, 3, 1],
                ]
            ],
            dtype=np.float32,
        ),
        ignored_boxes=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ignored_labels=np.array([0], dtype=np.int64),
        filename="synthetic.jpg",
        original_shape=image.shape,
        image_shape=image.shape,
        pad_shape=image.shape,
    )


def test_resize_scales_boxes_and_keypoints() -> None:
    sample = Resize((12, 8), keep_ratio=False)(_sample())

    assert sample.image.shape == (8, 12, 3)
    np.testing.assert_allclose(sample.boxes, [[2, 2, 8, 6]])
    np.testing.assert_allclose(sample.keypoints[0, :, :2], [[2, 2], [8, 2], [4, 4], [2, 6], [8, 6]])


def test_horizontal_flip_keeps_legacy_keypoint_order() -> None:
    sample = RandomHorizontalFlip(1.0)(_sample())

    np.testing.assert_allclose(sample.boxes, [[2, 1, 5, 3]])
    np.testing.assert_allclose(sample.keypoints[0, :, :2], [[2, 1], [5, 1], [4, 2], [2, 3], [5, 3]])
    assert sample.flip is True
    assert sample.flip_direction == "horizontal"


def test_random_square_crop_can_pad_outside_image() -> None:
    np.random.seed(0)
    sample = FaceSample(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        boxes=np.array([[0, 0, 4, 4]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        keypoints=np.array([[[0, 0, 1], [4, 0, 1], [2, 2, 1], [0, 4, 1], [4, 4, 1]]], dtype=np.float32),
        ignored_boxes=np.zeros((0, 4), dtype=np.float32),
        ignored_labels=np.zeros((0,), dtype=np.int64),
        filename="synthetic.jpg",
        original_shape=(4, 4, 3),
        image_shape=(4, 4, 3),
        pad_shape=(4, 4, 3),
    )

    cropped = RandomSquareCrop(crop_choice=(1.5,))(sample)

    assert cropped.image.shape == (6, 6, 3)
    assert np.any(cropped.image == 128)
    assert cropped.boxes.shape == (1, 4)


def test_filter_small_boxes_removes_aligned_targets() -> None:
    sample = FaceSample(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        boxes=np.array([[0, 0, 4, 4], [2, 2, 14, 15]], dtype=np.float32),
        labels=np.array([0, 0], dtype=np.int64),
        keypoints=np.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[3, 3, 1], [12, 3, 1], [8, 8, 1], [4, 14, 1], [12, 14, 1]],
            ],
            dtype=np.float32,
        ),
        ignored_boxes=np.array([[0, 0, 3, 3], [0, 0, 10, 10]], dtype=np.float32),
        ignored_labels=np.array([0, 0], dtype=np.int64),
        filename="synthetic.jpg",
        original_shape=(16, 16, 3),
        image_shape=(16, 16, 3),
        pad_shape=(16, 16, 3),
    )

    filtered = FilterSmallBoxes(min_size=10)(sample)

    np.testing.assert_allclose(filtered.boxes, [[2, 2, 14, 15]])
    assert filtered.labels.tolist() == [0]
    assert filtered.keypoints.shape == (1, 5, 3)
    np.testing.assert_allclose(filtered.ignored_boxes, [[0, 0, 10, 10]])


def test_to_tensor_and_collate_face_samples() -> None:
    sample = ToTensor()(Pad(size=(8, 8))(_sample()))
    batch = collate_face_samples([sample, sample])

    assert tuple(batch.images.shape) == (2, 3, 8, 8)
    assert all(isinstance(boxes, torch.Tensor) for boxes in batch.boxes)
    assert batch.boxes[0].shape == (1, 4)
    assert batch.metas[0]["filename"] == "synthetic.jpg"


def test_widerface_dataset_loads_image_and_applies_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    import yunet_train.tasks.face.dataset as dataset_module

    def fake_imread(path: str, flags: int) -> np.ndarray:
        normalized = path.replace("\\", "/")
        assert normalized.endswith("0--Parade/0_Parade_marchingband_1_849.jpg")
        return np.zeros((1385, 1024, 3), dtype=np.uint8)

    monkeypatch.setattr(dataset_module.cv2, "imread", fake_imread)
    dataset = WIDERFaceDataset(
        ann_file=Path(__file__).resolve().parents[1] / "data" / "widerface" / "labelv2" / "train" / "labelv2.txt",
        img_prefix=Path(__file__).resolve().parents[1] / "data" / "widerface" / "WIDER_train" / "images",
        transform=ToTensor(),
    )

    sample = dataset[0]
    assert isinstance(sample.image, torch.Tensor)
    assert tuple(sample.image.shape) == (3, 1385, 1024)
    assert sample.boxes.shape == (1, 4)
