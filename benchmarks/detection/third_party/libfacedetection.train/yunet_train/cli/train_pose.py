from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from yunet_train.tasks.pose import (
    COCO17_FLIP_IDX,
    COCO8_POSE_ROOT,
    CocoJsonPoseDataset,
    YOLOPoseDataset,
    YuNetPoseCriterion,
    build_pose_eval_transforms,
    build_pose_train_transforms,
    build_yunet_pose,
    collate_pose_samples,
    evaluate_pose_loss,
    train_pose_one_epoch,
)
from yunet_train.engine import (
    LinearWarmupMultiStepLR,
    load_checkpoint,
    load_model_weights_only,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YuNet pose detector.")
    parser.add_argument("--data-root", type=Path, default=COCO8_POSE_ROOT)
    parser.add_argument("--data-format", choices=("yolo", "coco_json"), default="yolo")
    parser.add_argument("--coco-train-ann", type=Path, default=None, help="COCO keypoints JSON for training split.")
    parser.add_argument(
        "--coco-train-images", type=Path, default=None, help="Directory of training images (e.g. .../train2017)."
    )
    parser.add_argument("--coco-val-ann", type=Path, default=None)
    parser.add_argument("--coco-val-images", type=Path, default=None)
    parser.add_argument("--variant", default="yunet_n", choices=("yunet_n", "yunet_s"))
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/yunet_pose"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-steps", type=int, nargs="*", default=[48, 56])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--warmup-ratio", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="With --resume: load model weights only; reset optimizer, LR scheduler, and epoch counter (for domain fine-tuning).",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--eval-limit-samples", type=int, default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--min-box-size", type=float, default=10.0)


def main() -> None:
    run_training(parse_args())


def run_training(args: argparse.Namespace) -> None:
    _validate_pose_train_args(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.log_file or args.work_dir / "train_pose.log")
    try:
        _log_header(logger, args)
        device = torch.device(args.device)
        model = build_yunet_pose(args.variant, kpt_shape=(17, 3)).to(device)
        criterion = YuNetPoseCriterion(strides=(8, 16, 32), kpt_shape=(17, 3))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        lr_scheduler = LinearWarmupMultiStepLR(
            optimizer,
            milestones=tuple(args.lr_steps),
            gamma=args.lr_gamma,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
        )

        start_epoch = 1
        best_loss = _read_best_loss(args.work_dir)
        if args.resume is not None:
            if getattr(args, "resume_weights_only", False):
                load_model_weights_only(args.resume, model=model, map_location="cpu")
                model.to(device)
                start_epoch = 1
                best_loss = None
                logger(
                    f"resume_weights_only path={args.resume} "
                    f"optimizer_and_scheduler_reset start_epoch={start_epoch}"
                )
            else:
                checkpoint = load_checkpoint(
                    args.resume,
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    map_location="cpu",
                )
                _move_optimizer_state_to_device(optimizer, device)
                start_epoch = int(checkpoint.get("epoch", 0)) + 1
                best_loss = _checkpoint_best_loss(checkpoint, fallback=best_loss)
                logger(
                    f"resumed_checkpoint path={args.resume} "
                    f"start_epoch={start_epoch} "
                    f"best_loss={best_loss if best_loss is not None else 'none'}"
                )

        train_loader = _build_loader(args, split="train", shuffle=True, device=device)
        val_loader = _build_val_loader(args, device=device)
        logger(f"train_loader steps={len(train_loader)} batch_size={args.batch_size} workers={args.workers}")
        if val_loader is not None:
            logger(f"val_loader steps={len(val_loader)} eval_interval={args.eval_interval}")
        elif args.eval_interval > 0 and getattr(args, "data_format", "yolo") == "coco_json":
            logger("coco_json: eval_interval>0 but no val annotations; validation disabled")

        started_at = time.perf_counter()
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started_at = time.perf_counter()
            stats = train_pose_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                log_interval=args.log_interval,
                logger=logger,
            )
            lr = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.perf_counter() - epoch_started_at
            logger(
                f"epoch={epoch}/{args.epochs} lr={lr:.8f} loss={stats.loss:.6f} "
                f"cls={stats.loss_cls:.6f} bbox={stats.loss_bbox:.6f} obj={stats.loss_obj:.6f} "
                f"kpt={stats.loss_kpt:.6f} kpt_vis={stats.loss_kpt_vis:.6f} "
                f"epoch_seconds={epoch_seconds:.3f}"
            )
            _try_append_metrics_csv(args.work_dir / "metrics.csv", epoch, stats, lr=lr, logger=logger)
            _try_save_pose_checkpoint(
                path=args.work_dir / "latest.pth",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                args=args,
                stats=stats,
                lr=lr,
                extra_metrics={"best_loss": best_loss},
                logger=logger,
            )

            if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
                path = args.work_dir / f"epoch_{epoch}.pth"
                _try_save_pose_checkpoint(
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    args=args,
                    stats=stats,
                    lr=lr,
                    extra_metrics={"best_loss": best_loss},
                    logger=logger,
                )

            if val_loader is not None and epoch % args.eval_interval == 0:
                try:
                    val_stats = evaluate_pose_loss(model=model, criterion=criterion, data_loader=val_loader, device=device)
                    logger(
                        f"eval epoch={epoch} loss={val_stats.loss:.6f} cls={val_stats.loss_cls:.6f} "
                        f"bbox={val_stats.loss_bbox:.6f} obj={val_stats.loss_obj:.6f} "
                        f"kpt={val_stats.loss_kpt:.6f} kpt_vis={val_stats.loss_kpt_vis:.6f}"
                    )
                    _try_append_metrics_csv(args.work_dir / "val_metrics.csv", epoch, val_stats, lr=lr, logger=logger)
                    _try_save_pose_checkpoint(
                        path=args.work_dir / f"eval_epoch_{epoch}.pth",
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        args=args,
                        stats=val_stats,
                        lr=lr,
                        extra_metrics={"best_loss": best_loss},
                        logger=logger,
                    )
                    best_loss = _maybe_save_best_checkpoint(
                        work_dir=args.work_dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        args=args,
                        stats=val_stats,
                        lr=lr,
                        best_loss=best_loss,
                        logger=logger,
                    )
                except Exception as exc:
                    logger(f"eval_failed epoch={epoch} error={type(exc).__name__}: {exc}")
            elif val_loader is None:
                best_loss = _maybe_save_best_checkpoint(
                    work_dir=args.work_dir,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    args=args,
                    stats=stats,
                    lr=lr,
                    best_loss=best_loss,
                    logger=logger,
                )
        logger(f"run_finished elapsed_seconds={time.perf_counter() - started_at:.3f}")
    finally:
        logger.close()


def _build_val_loader(args: argparse.Namespace, *, device: torch.device) -> DataLoader | None:
    if args.eval_interval <= 0:
        return None
    if getattr(args, "data_format", "yolo") == "coco_json" and args.coco_val_ann is None:
        return None
    return _build_loader(args, split="val", shuffle=False, device=device)


def _validate_pose_train_args(args: argparse.Namespace) -> None:
    if getattr(args, "resume_weights_only", False) and args.resume is None:
        raise ValueError("--resume-weights-only requires --resume")
    if getattr(args, "data_format", "yolo") == "coco_json":
        if args.coco_train_ann is None or args.coco_train_images is None:
            raise ValueError("data-format=coco_json requires --coco-train-ann and --coco-train-images")
        ca, ci = args.coco_val_ann, args.coco_val_images
        if (ca is None) != (ci is None):
            raise ValueError("Provide both --coco-val-ann and --coco-val-images, or neither")


def _build_pose_dataset(
    args: argparse.Namespace,
    *,
    split: str,
    transform: Any,
) -> Dataset:
    fmt = getattr(args, "data_format", "yolo")
    if fmt == "yolo":
        return YOLOPoseDataset(
            args.data_root,
            split=split,
            transform=transform,
            kpt_shape=(17, 3),
        )
    if split == "train":
        return CocoJsonPoseDataset(
            args.coco_train_ann,
            args.coco_train_images,
            transform=transform,
            kpt_shape=(17, 3),
        )
    assert args.coco_val_ann is not None and args.coco_val_images is not None
    return CocoJsonPoseDataset(
        args.coco_val_ann,
        args.coco_val_images,
        transform=transform,
        kpt_shape=(17, 3),
    )


def _build_loader(args: argparse.Namespace, *, split: str, shuffle: bool, device: torch.device) -> DataLoader:
    transform = (
        build_pose_train_transforms(
            args.image_size,
            flip_idx=COCO17_FLIP_IDX,
            random_crop=not getattr(args, "no_random_crop", False),
            min_box_size=getattr(args, "min_box_size", 10.0),
        )
        if split == "train"
        else build_pose_eval_transforms(args.image_size)
    )
    dataset = _build_pose_dataset(args, split=split, transform=transform)
    limit = args.limit_samples if split == "train" else args.eval_limit_samples
    if limit is not None:
        dataset.records = dataset.records[:limit]
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.workers,
        "collate_fn": collate_pose_samples,
        "pin_memory": device.type == "cuda" and not args.no_pin_memory,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["worker_init_fn"] = _init_worker
    return DataLoader(dataset, **kwargs)


def _init_worker(worker_id: int) -> None:
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _save_pose_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LinearWarmupMultiStepLR,
    epoch: int,
    args: argparse.Namespace,
    stats: Any,
    lr: float,
    extra_metrics: dict[str, float | None] | None = None,
) -> None:
    metrics = {
        "loss": stats.loss,
        "loss_cls": stats.loss_cls,
        "loss_bbox": stats.loss_bbox,
        "loss_obj": stats.loss_obj,
        "loss_kpt": stats.loss_kpt,
        "loss_kpt_vis": stats.loss_kpt_vis,
        "lr": lr,
    }
    if extra_metrics is not None:
        metrics.update({key: value for key, value in extra_metrics.items() if value is not None})
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        config=_serializable_config(args),
        metrics=metrics,
        scheduler_state=lr_scheduler.state_dict(),
    )


def _try_save_pose_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LinearWarmupMultiStepLR,
    epoch: int,
    args: argparse.Namespace,
    stats: Any,
    lr: float,
    extra_metrics: dict[str, float | None] | None,
    logger: "RunLogger",
) -> bool:
    try:
        _save_pose_checkpoint(path, model, optimizer, lr_scheduler, epoch, args, stats, lr, extra_metrics=extra_metrics)
    except Exception as exc:
        logger(f"checkpoint_save_failed path={path} error={type(exc).__name__}: {exc}")
        return False
    logger(f"saved_checkpoint path={path}")
    return True


def _maybe_save_best_checkpoint(
    *,
    work_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LinearWarmupMultiStepLR,
    epoch: int,
    args: argparse.Namespace,
    stats: Any,
    lr: float,
    best_loss: float | None,
    logger: "RunLogger",
) -> float:
    if best_loss is not None and stats.loss >= best_loss:
        return best_loss

    candidate_loss = stats.loss
    best_path = work_dir / "best_loss.pth"
    try:
        _save_pose_checkpoint(
            best_path,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            args,
            stats,
            lr,
            extra_metrics={"best_loss": candidate_loss},
        )
        _write_best_loss(work_dir, best_loss=candidate_loss, epoch=epoch)
    except Exception as exc:
        logger(f"best_checkpoint_save_failed path={best_path} error={type(exc).__name__}: {exc}")
        return best_loss
    logger(f"saved_best_checkpoint path={best_path} best_loss={candidate_loss:.6f}")
    return candidate_loss


def _append_metrics_csv(path: Path, epoch: int, stats: Any, *, lr: float) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "lr", "loss", "loss_cls", "loss_bbox", "loss_obj", "loss_kpt", "loss_kpt_vis", "steps"),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "lr": lr,
                "loss": stats.loss,
                "loss_cls": stats.loss_cls,
                "loss_bbox": stats.loss_bbox,
                "loss_obj": stats.loss_obj,
                "loss_kpt": stats.loss_kpt,
                "loss_kpt_vis": stats.loss_kpt_vis,
                "steps": stats.steps,
            }
        )


def _try_append_metrics_csv(path: Path, epoch: int, stats: Any, *, lr: float, logger: "RunLogger") -> bool:
    try:
        _append_metrics_csv(path, epoch, stats, lr=lr)
    except Exception as exc:
        logger(f"metrics_write_failed path={path} error={type(exc).__name__}: {exc}")
        return False
    return True


def _serializable_config(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _read_best_loss(work_dir: Path) -> float | None:
    best_file = work_dir / "best_loss.txt"
    if not best_file.exists():
        return None
    try:
        first_line = best_file.read_text(encoding="utf-8").splitlines()[0]
        return float(first_line.split(",", maxsplit=1)[0])
    except (IndexError, OSError, ValueError):
        return None


def _write_best_loss(work_dir: Path, *, best_loss: float, epoch: int) -> None:
    (work_dir / "best_loss.txt").write_text(f"{best_loss:.12g},{epoch}\n", encoding="utf-8")


def _checkpoint_best_loss(checkpoint: dict[str, Any], *, fallback: float | None) -> float | None:
    metrics = checkpoint.get("metrics", {})
    if isinstance(metrics, dict) and "best_loss" in metrics:
        return float(metrics["best_loss"])
    return fallback


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _log_header(logger: "RunLogger", args: argparse.Namespace) -> None:
    logger("=" * 80)
    logger(f"run_started_at={datetime.now():%Y-%m-%d %H:%M:%S}")
    logger(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
    for key, value in sorted(vars(args).items()):
        logger(f"arg.{key}={value}")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        print(line, flush=True)
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except OSError as exc:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] log_write_failed error={type(exc).__name__}: {exc}", flush=True)

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
