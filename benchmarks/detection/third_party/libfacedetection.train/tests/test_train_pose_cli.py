from __future__ import annotations

import argparse
from pathlib import Path

from yunet_train.tasks.pose import COCO8_POSE_ROOT
from shutil import rmtree

import pytest
import torch

import yunet_train.cli.train_pose as train_pose_cli


def _coco8_pose_root() -> Path:
    return COCO8_POSE_ROOT


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_train_pose_cli_smoke_saves_checkpoint() -> None:
    work_dir = Path(__file__).resolve().parent / "output" / "train_pose_cli_smoke"
    if work_dir.exists():
        rmtree(work_dir)
    args = argparse.Namespace(
        data_root=_coco8_pose_root(),
        data_format="yolo",
        coco_train_ann=None,
        coco_train_images=None,
        coco_val_ann=None,
        coco_val_images=None,
        variant="yunet_n",
        work_dir=work_dir,
        image_size=64,
        batch_size=1,
        workers=0,
        prefetch_factor=1,
        epochs=1,
        lr=1e-4,
        lr_steps=[400, 544],
        lr_gamma=0.1,
        warmup_iters=0,
        warmup_ratio=0.001,
        momentum=0.9,
        weight_decay=0.0,
        device="cpu",
        checkpoint_interval=1,
        eval_interval=1,
        resume=None,
        resume_weights_only=False,
        limit_samples=1,
        eval_limit_samples=1,
        no_pin_memory=False,
        no_persistent_workers=False,
        log_interval=1,
        log_file=None,
        no_random_crop=False,
        min_box_size=10.0,
    )

    train_pose_cli.run_training(args)

    latest_checkpoint = work_dir / "latest.pth"
    epoch_checkpoint = work_dir / "epoch_1.pth"
    eval_checkpoint = work_dir / "eval_epoch_1.pth"
    best_checkpoint = work_dir / "best_loss.pth"
    best_loss_file = work_dir / "best_loss.txt"
    assert latest_checkpoint.exists()
    assert epoch_checkpoint.exists()
    assert eval_checkpoint.exists()
    assert best_checkpoint.exists()
    assert best_loss_file.exists()
    assert (work_dir / "metrics.csv").exists()
    assert (work_dir / "val_metrics.csv").exists()
    assert (work_dir / "train_pose.log").exists()
    latest_data = torch.load(latest_checkpoint, map_location="cpu")
    best_data = torch.load(best_checkpoint, map_location="cpu")
    assert latest_data["epoch"] == 1
    assert best_data["epoch"] == 1
    assert "best_loss" in best_data["metrics"]
    assert "loss_kpt_vis" in latest_data["metrics"]
    assert "loss_kpt_vis" in (work_dir / "metrics.csv").read_text(encoding="utf-8")
    rmtree(work_dir)


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_train_pose_cli_resumes_from_latest_checkpoint() -> None:
    work_dir = Path(__file__).resolve().parent / "output" / "train_pose_cli_resume"
    if work_dir.exists():
        rmtree(work_dir)
    args = argparse.Namespace(
        data_root=_coco8_pose_root(),
        data_format="yolo",
        coco_train_ann=None,
        coco_train_images=None,
        coco_val_ann=None,
        coco_val_images=None,
        variant="yunet_n",
        work_dir=work_dir,
        image_size=64,
        batch_size=1,
        workers=0,
        prefetch_factor=1,
        epochs=1,
        lr=1e-4,
        lr_steps=[400, 544],
        lr_gamma=0.1,
        warmup_iters=0,
        warmup_ratio=0.001,
        momentum=0.9,
        weight_decay=0.0,
        device="cpu",
        checkpoint_interval=10,
        eval_interval=0,
        resume=None,
        resume_weights_only=False,
        limit_samples=1,
        eval_limit_samples=None,
        no_pin_memory=False,
        no_persistent_workers=False,
        log_interval=0,
        log_file=None,
        no_random_crop=False,
        min_box_size=10.0,
    )
    train_pose_cli.run_training(args)

    args.resume = work_dir / "latest.pth"
    args.epochs = 2
    train_pose_cli.run_training(args)

    latest_data = torch.load(work_dir / "latest.pth", map_location="cpu")
    best_data = torch.load(work_dir / "best_loss.pth", map_location="cpu")
    log_text = (work_dir / "train_pose.log").read_text(encoding="utf-8")
    assert latest_data["epoch"] == 2
    assert best_data["epoch"] in {1, 2}
    assert "best_loss" in best_data["metrics"]
    assert "resumed_checkpoint" in log_text
    rmtree(work_dir)


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_train_pose_cli_checkpoint_interval_zero_disables_epoch_checkpoint() -> None:
    work_dir = Path(__file__).resolve().parent / "output" / "train_pose_cli_no_epoch_checkpoint"
    if work_dir.exists():
        rmtree(work_dir)
    args = argparse.Namespace(
        data_root=_coco8_pose_root(),
        data_format="yolo",
        coco_train_ann=None,
        coco_train_images=None,
        coco_val_ann=None,
        coco_val_images=None,
        variant="yunet_n",
        work_dir=work_dir,
        image_size=64,
        batch_size=1,
        workers=0,
        prefetch_factor=1,
        epochs=1,
        lr=1e-4,
        lr_steps=[400, 544],
        lr_gamma=0.1,
        warmup_iters=0,
        warmup_ratio=0.001,
        momentum=0.9,
        weight_decay=0.0,
        device="cpu",
        checkpoint_interval=0,
        eval_interval=0,
        resume=None,
        resume_weights_only=False,
        limit_samples=1,
        eval_limit_samples=None,
        no_pin_memory=False,
        no_persistent_workers=False,
        log_interval=0,
        log_file=None,
        no_random_crop=False,
        min_box_size=10.0,
    )

    train_pose_cli.run_training(args)

    assert (work_dir / "latest.pth").exists()
    assert (work_dir / "best_loss.pth").exists()
    assert not (work_dir / "epoch_1.pth").exists()
    rmtree(work_dir)
