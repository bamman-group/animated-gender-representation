from __future__ import annotations

import argparse
from pathlib import Path

from yunet_train.tasks.pose import COCO8_POSE_ROOT
from shutil import rmtree

import pytest
import torch

from yunet_train.cli.eval_pose import eval_pose
from yunet_train.tasks.pose import build_yunet_pose


def _coco8_pose_root() -> Path:
    return COCO8_POSE_ROOT


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_eval_pose_cli_smoke_writes_metrics_and_visualization() -> None:
    work_dir = Path(__file__).resolve().parent / "output" / "eval_pose_cli"
    if work_dir.exists():
        rmtree(work_dir)
    work_dir.mkdir(parents=True)
    checkpoint = _checkpoint(work_dir / "yunet_pose_n.pth")

    stats = eval_pose(
        argparse.Namespace(
            checkpoint=checkpoint,
            data_root=_coco8_pose_root(),
            variant="yunet_n",
            image_size=64,
            batch_size=1,
            workers=0,
            device="cpu",
            limit_samples=1,
            out_dir=work_dir,
            save_visualizations=1,
            score_threshold=0.0,
            nms_threshold=0.45,
        )
    )

    assert stats.loss > 0
    assert (work_dir / "pose_eval_metrics.csv").exists()
    assert list((work_dir / "visualizations").glob("*.jpg"))
    rmtree(work_dir)


def _checkpoint(path: Path) -> Path:
    model = build_yunet_pose("yunet_n", kpt_shape=(17, 3))
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "config": {"variant": "yunet_n"},
            "metrics": {},
        },
        path,
    )
    return path
