from __future__ import annotations

import argparse
from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest

from yunet_train.cli.eval_widerface import run_evaluation
from yunet_train.tasks.face import WIDER_VAL_ANN_FILE, WIDER_VAL_GT_DIR, WIDER_VAL_IMAGE_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_yunet_n_checkpoint_runs_widerface_eval_smoke() -> None:
    checkpoint = REPO_ROOT / "weights" / "yunet_n.pth"
    gt_dir = WIDER_VAL_GT_DIR
    if not checkpoint.exists():
        pytest.skip(f"legacy checkpoint not found: {checkpoint}")
    if not gt_dir.exists():
        pytest.skip(f"WIDERFace val gt dir not found: {gt_dir}")
    expected_image = WIDER_VAL_IMAGE_DIR / "0--Parade" / "0_Parade_marchingband_1_465.jpg"
    if not expected_image.exists():
        pytest.skip(f"WIDERFace val image not found: {expected_image}")

    output_dir = Path(__file__).resolve().parent / "output" / "legacy_yunet_n_widerface_eval"
    if output_dir.exists():
        rmtree(output_dir)

    aps = run_evaluation(
        argparse.Namespace(
            checkpoint=checkpoint,
            variant="yunet_n",
            ann_file=WIDER_VAL_ANN_FILE,
            img_prefix=WIDER_VAL_IMAGE_DIR,
            gt_dir=gt_dir,
            output_dir=output_dir,
            mode="origin",
            image_size=640,
            size_divisor=32,
            batch_size=1,
            workers=0,
            device="cpu",
            score_threshold=0.02,
            nms_threshold=0.45,
            max_detections=-1,
            iou_threshold=0.5,
            limit_samples=1,
            save_preds=True,
        )
    )

    assert np.isfinite([aps.easy, aps.medium, aps.hard]).all()
    assert (output_dir / "aps.txt").exists()
    prediction_file = output_dir / "predictions" / "0--Parade" / "0_Parade_marchingband_1_465.txt"
    lines = prediction_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0--Parade/0_Parade_marchingband_1_465.jpg"
    assert int(lines[1]) > 0
    assert len(lines) == int(lines[1]) + 2
