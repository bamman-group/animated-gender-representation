from __future__ import annotations

import argparse
from pathlib import Path

from yunet_train.tasks.pose import COCO8_POSE_ROOT
from shutil import rmtree

import numpy as np
import pytest

from yunet_train.tasks.pose import PoseSample, pose_sample_annotation_text, render_pose_sample
from yunet_train.tools.visualize_pose_dataset import visualize_pose_dataset


def _coco8_pose_root() -> Path:
    return COCO8_POSE_ROOT


def _sample() -> PoseSample:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, :, 0] = np.linspace(4, 28, 17, dtype=np.float32)
    keypoints[0, :, 1] = np.linspace(4, 28, 17, dtype=np.float32)
    keypoints[0, :, 2] = 2
    return PoseSample(
        image=image,
        boxes=np.array([[2, 3, 30, 31]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        keypoints=keypoints,
        filename="synthetic.jpg",
        original_shape=image.shape,
        image_shape=image.shape,
        pad_shape=image.shape,
        kpt_shape=(17, 3),
    )


def test_render_pose_sample_draws_on_image() -> None:
    rendered = render_pose_sample(_sample())

    assert rendered.shape == (32, 32, 3)
    assert rendered.dtype == np.uint8
    assert rendered.sum() > 0


def test_pose_sample_annotation_text_contains_visibility_summary() -> None:
    text = pose_sample_annotation_text(_sample())

    assert "objects: 1" in text
    assert "visible_keypoints=17" in text


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_visualize_pose_dataset_writes_debug_files() -> None:
    out_dir = Path(__file__).resolve().parent / "output" / "pose_visualize_smoke"
    if out_dir.exists():
        rmtree(out_dir)
    args = argparse.Namespace(
        data_root=_coco8_pose_root(),
        split="train",
        image_size=64,
        epochs=1,
        samples_per_epoch=1,
        out_dir=out_dir,
        seed=0,
        no_transform=False,
    )

    visualize_pose_dataset(args)

    assert (out_dir / "epoch_001_sample_00.jpg").exists()
    assert (out_dir / "epoch_001_sample_00.txt").exists()
    rmtree(out_dir)
