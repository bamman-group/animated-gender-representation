from __future__ import annotations

import argparse
from pathlib import Path

from yunet_train.tasks.pose import COCO8_POSE_ROOT
from shutil import rmtree

import pytest

from yunet_train.tools.check_pose_overfit import run_overfit_check


def _coco8_pose_root() -> Path:
    return COCO8_POSE_ROOT


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_pose_overfit_check_detects_loss_decrease() -> None:
    work_dir = Path(__file__).resolve().parent / "output" / "pose_overfit_check"
    if work_dir.exists():
        rmtree(work_dir)
    args = argparse.Namespace(
        data_root=_coco8_pose_root(),
        variant="yunet_n",
        work_dir=work_dir,
        image_size=64,
        samples=1,
        epochs=3,
        batch_size=1,
        workers=0,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
        seed=0,
        min_loss_ratio=0.999,
        augment=False,
        no_save_checkpoint=True,
    )

    result = run_overfit_check(args)

    assert result.passed
    assert result.best_loss < result.initial_loss
    assert (work_dir / "overfit_metrics.csv").exists()
    assert (work_dir / "overfit_summary.json").exists()
    assert "PASS" in (work_dir / "overfit_summary.txt").read_text(encoding="utf-8")
    rmtree(work_dir)
