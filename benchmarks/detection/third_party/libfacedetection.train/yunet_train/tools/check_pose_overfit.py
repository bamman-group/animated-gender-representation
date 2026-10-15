from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from yunet_train.tasks.pose import (
    COCO17_FLIP_IDX,
    COCO8_POSE_ROOT,
    YOLOPoseDataset,
    YuNetPoseCriterion,
    build_pose_eval_transforms,
    build_pose_train_transforms,
    build_yunet_pose,
    collate_pose_samples,
    evaluate_pose_loss,
    train_pose_one_epoch,
)
from yunet_train.tasks.pose.trainer import PoseTrainStats
from yunet_train.engine import save_checkpoint


@dataclass(frozen=True)
class OverfitCheckResult:
    passed: bool
    initial_loss: float
    final_loss: float
    best_loss: float
    best_epoch: int
    required_best_loss: float
    epochs: int
    samples: int
    image_size: int
    work_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether YuNet pose can overfit a tiny YOLO-pose subset.")
    parser.add_argument("--data-root", type=Path, default=COCO8_POSE_ROOT)
    parser.add_argument("--variant", default="yunet_n", choices=("yunet_n", "yunet_s"))
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/pose_overfit_check"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-loss-ratio", type=float, default=0.8)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_overfit_check(parse_args())
    if not result.passed:
        raise SystemExit(1)


def run_overfit_check(args: argparse.Namespace) -> OverfitCheckResult:
    if not 0.0 < args.min_loss_ratio < 1.0:
        raise ValueError("min_loss_ratio must be in (0, 1)")
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    _set_seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pose_samples,
        pin_memory=device.type == "cuda",
    )
    if len(loader) == 0:
        raise ValueError("overfit data loader has no batches")

    model = build_yunet_pose(args.variant, kpt_shape=(17, 3)).to(device)
    criterion = YuNetPoseCriterion(strides=(8, 16, 32), kpt_shape=(17, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_path = args.work_dir / "overfit_metrics.csv"
    initial_stats = evaluate_pose_loss(model=model, criterion=criterion, data_loader=loader, device=device)
    rows: list[dict[str, float | int]] = [_row_from_stats(0, initial_stats, train_stats=None)]
    _write_rows(metrics_path, rows)

    best_loss = initial_stats.loss
    best_epoch = 0
    final_stats = initial_stats
    for epoch in range(1, args.epochs + 1):
        train_stats = train_pose_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        final_stats = evaluate_pose_loss(model=model, criterion=criterion, data_loader=loader, device=device)
        if final_stats.loss < best_loss:
            best_loss = final_stats.loss
            best_epoch = epoch
        rows.append(_row_from_stats(epoch, final_stats, train_stats=train_stats))
        _write_rows(metrics_path, rows)

    required_best_loss = initial_stats.loss * args.min_loss_ratio
    passed = best_loss <= required_best_loss and torch.isfinite(torch.tensor(final_stats.loss)).item()
    result = OverfitCheckResult(
        passed=passed,
        initial_loss=initial_stats.loss,
        final_loss=final_stats.loss,
        best_loss=best_loss,
        best_epoch=best_epoch,
        required_best_loss=required_best_loss,
        epochs=args.epochs,
        samples=len(dataset),
        image_size=args.image_size,
        work_dir=str(args.work_dir),
    )
    _write_summary(args.work_dir / "overfit_summary.json", result)
    _write_summary_text(args.work_dir / "overfit_summary.txt", result)
    if not args.no_save_checkpoint:
        save_checkpoint(
            path=args.work_dir / "overfit_latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            config=_serializable_config(args),
            metrics=asdict(result),
        )
    _print_result(result)
    return result


def _build_dataset(args: argparse.Namespace) -> YOLOPoseDataset:
    transform = (
        build_pose_train_transforms(args.image_size, flip_idx=COCO17_FLIP_IDX, random_crop=False)
        if args.augment
        else build_pose_eval_transforms(args.image_size)
    )
    dataset = YOLOPoseDataset(args.data_root, split="train", transform=transform, kpt_shape=(17, 3))
    dataset.records = dataset.records[: args.samples]
    if len(dataset) == 0:
        raise ValueError(f"No pose samples found under {args.data_root}")
    return dataset


def _row_from_stats(
    epoch: int,
    eval_stats: PoseTrainStats,
    *,
    train_stats: PoseTrainStats | None,
) -> dict[str, float | int | str]:
    return {
        "epoch": epoch,
        "train_loss": "" if train_stats is None else train_stats.loss,
        "eval_loss": eval_stats.loss,
        "eval_loss_cls": eval_stats.loss_cls,
        "eval_loss_bbox": eval_stats.loss_bbox,
        "eval_loss_obj": eval_stats.loss_obj,
        "eval_loss_kpt": eval_stats.loss_kpt,
        "eval_loss_kpt_vis": eval_stats.loss_kpt_vis,
        "steps": eval_stats.steps,
    }


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, result: OverfitCheckResult) -> None:
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def _write_summary_text(path: Path, result: OverfitCheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    path.write_text(
        "\n".join(
            [
                f"status: {status}",
                f"initial_loss: {result.initial_loss:.6f}",
                f"final_loss: {result.final_loss:.6f}",
                f"best_loss: {result.best_loss:.6f}",
                f"best_epoch: {result.best_epoch}",
                f"required_best_loss: {result.required_best_loss:.6f}",
                f"epochs: {result.epochs}",
                f"samples: {result.samples}",
                f"image_size: {result.image_size}",
                f"work_dir: {result.work_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _print_result(result: OverfitCheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status} initial_loss={result.initial_loss:.6f} "
        f"final_loss={result.final_loss:.6f} "
        f"best_loss={result.best_loss:.6f} "
        f"required_best_loss={result.required_best_loss:.6f} "
        f"best_epoch={result.best_epoch}"
    )


def _set_seed(seed: int) -> None:
    cv2.setNumThreads(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


if __name__ == "__main__":
    main()
