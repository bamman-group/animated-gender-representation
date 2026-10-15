from __future__ import annotations

import argparse
from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
import torch

import yunet_train.cli.train as train_cli
from yunet_train.tasks.face import FaceSample, ToTensor


class SyntheticDataset:
    def __init__(self, *args: object, transform: object | None = None, **kwargs: object):
        self.transform = transform
        self.records = [object()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FaceSample:
        del index
        sample = FaceSample(
            image=np.ones((64, 64, 3), dtype=np.float32),
            boxes=np.array([[0, 0, 8, 8]], dtype=np.float32),
            labels=np.array([0], dtype=np.int64),
            keypoints=np.array(
                [
                    [
                        [0, 0, 1],
                        [8, 0, 1],
                        [4, 4, 1],
                        [0, 8, 1],
                        [8, 8, 1],
                    ]
                ],
                dtype=np.float32,
            ),
            ignored_boxes=np.zeros((0, 4), dtype=np.float32),
            ignored_labels=np.zeros((0,), dtype=np.int64),
            filename="synthetic.jpg",
            original_shape=(64, 64, 3),
            image_shape=(64, 64, 3),
            pad_shape=(64, 64, 3),
        )
        return ToTensor()(sample)


def test_train_cli_smoke_saves_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_cli, "WIDERFaceDataset", SyntheticDataset)
    monkeypatch.setattr(train_cli, "build_train_transforms", lambda **kwargs: None)
    monkeypatch.setattr(train_cli, "build_eval_transforms", lambda **kwargs: None)
    work_dir = Path(__file__).resolve().parent / "output" / "train_cli_smoke"
    if work_dir.exists():
        rmtree(work_dir)
    args = argparse.Namespace(
        variant="yunet_s",
        ann_file=Path("unused.txt"),
        img_prefix=Path("unused"),
        val_ann_file=Path("unused_val.txt"),
        val_img_prefix=Path("unused_val"),
        work_dir=work_dir,
        image_size=64,
        min_face_size=0.0,
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
        limit_samples=None,
        eval_limit_samples=None,
        no_tensorboard=True,
        no_pin_memory=False,
        no_persistent_workers=False,
        log_interval=0,
        log_file=None,
    )

    train_cli.run_training(args)

    checkpoint = work_dir / "epoch_1.pth"
    eval_checkpoint = work_dir / "eval_epoch_1.pth"
    latest_checkpoint = work_dir / "latest.pth"
    best_checkpoint = work_dir / "best_loss.pth"
    best_loss_file = work_dir / "best_loss.txt"
    metrics_file = work_dir / "metrics.csv"
    val_metrics_file = work_dir / "val_metrics.csv"
    log_file = work_dir / "train.log"
    assert checkpoint.exists()
    assert eval_checkpoint.exists()
    assert latest_checkpoint.exists()
    assert best_checkpoint.exists()
    assert best_loss_file.exists()
    assert metrics_file.exists()
    assert val_metrics_file.exists()
    assert log_file.exists()
    metrics_text = metrics_file.read_text(encoding="utf-8")
    assert "loss_cls" in metrics_text
    assert "lr" in metrics_text
    log_text = log_file.read_text(encoding="utf-8")
    assert "run_started_at" in log_text
    assert "train_dataset samples=1" in log_text
    assert "saved_checkpoint" in log_text
    assert "loss_cls" in val_metrics_file.read_text(encoding="utf-8")
    data = torch.load(checkpoint, map_location="cpu")
    latest_data = torch.load(latest_checkpoint, map_location="cpu")
    best_data = torch.load(best_checkpoint, map_location="cpu")
    eval_data = torch.load(eval_checkpoint, map_location="cpu")
    assert data["epoch"] == 1
    assert latest_data["epoch"] == 1
    assert best_data["epoch"] == 1
    assert "state_dict" in data
    assert "lr_scheduler" in data
    assert eval_data["metrics"]["val_loss"] > 0
    assert best_data["metrics"]["best_loss"] == eval_data["metrics"]["val_loss"]
    rmtree(work_dir)


def test_train_cli_resume_from_latest_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_cli, "WIDERFaceDataset", SyntheticDataset)
    monkeypatch.setattr(train_cli, "build_train_transforms", lambda **kwargs: None)
    monkeypatch.setattr(train_cli, "build_eval_transforms", lambda **kwargs: None)
    work_dir = Path(__file__).resolve().parent / "output" / "train_cli_resume"
    if work_dir.exists():
        rmtree(work_dir)

    args = argparse.Namespace(
        variant="yunet_s",
        ann_file=Path("unused.txt"),
        img_prefix=Path("unused"),
        val_ann_file=Path("unused_val.txt"),
        val_img_prefix=Path("unused_val"),
        work_dir=work_dir,
        image_size=64,
        min_face_size=0.0,
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
        limit_samples=None,
        eval_limit_samples=None,
        no_tensorboard=True,
        no_pin_memory=False,
        no_persistent_workers=False,
        log_interval=0,
        log_file=None,
    )
    train_cli.run_training(args)

    args.resume = work_dir / "latest.pth"
    args.epochs = 2
    train_cli.run_training(args)

    latest_data = torch.load(work_dir / "latest.pth", map_location="cpu")
    assert latest_data["epoch"] == 2
    assert latest_data["lr_scheduler"]["iter_count"] == 2
    log_text = (work_dir / "train.log").read_text(encoding="utf-8")
    assert "resumed_checkpoint" in log_text
    rmtree(work_dir)


def test_train_cli_module_import_has_no_circular_import() -> None:
    assert train_cli.parse_args is not None
