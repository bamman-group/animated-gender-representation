from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from yunet_train.engine import LinearWarmupMultiStepLR, load_checkpoint, save_checkpoint
from yunet_train.tasks.face import (
    WIDERFaceDataset,
    WIDER_TRAIN_ANN_FILE,
    WIDER_TRAIN_IMAGE_DIR,
    WIDER_VAL_ANN_FILE,
    WIDER_VAL_IMAGE_DIR,
    YuNetCriterion,
    build_eval_transforms,
    build_train_transforms,
    build_yunet,
    collate_face_samples,
    evaluate_loss,
    get_train_crop_choice,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight YuNet.")
    parser.add_argument("--variant", default="yunet_n", choices=("yunet_n", "yunet_s"))
    parser.add_argument("--ann-file", type=Path, default=WIDER_TRAIN_ANN_FILE)
    parser.add_argument("--img-prefix", type=Path, default=WIDER_TRAIN_IMAGE_DIR)
    parser.add_argument("--val-ann-file", type=Path, default=WIDER_VAL_ANN_FILE)
    parser.add_argument("--val-img-prefix", type=Path, default=WIDER_VAL_IMAGE_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/yunet"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--min-face-size", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-steps", type=int, nargs="*", default=[400, 544])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--warmup-ratio", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--eval-limit-samples", type=int, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


def run_training(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logger = _build_run_logger(args)
    _log_run_header(logger, args)
    device = torch.device(args.device)
    model = build_yunet(args.variant).to(device)
    criterion = YuNetCriterion(strides=(8, 16, 32))
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
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            map_location="cpu",
        )
        _move_optimizer_state_to_device(optimizer, device)
        resumed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        best_loss = _checkpoint_best_loss(checkpoint, fallback=best_loss)
        logger(
            f"resumed_checkpoint path={args.resume} "
            f"epoch={resumed_epoch} "
            f"start_epoch={start_epoch} "
            f"best_loss={best_loss if best_loss is not None else 'none'}"
        )

    dataset = WIDERFaceDataset(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        transform=build_train_transforms(
            image_size=args.image_size,
            crop_choice=get_train_crop_choice(args.variant),
            min_box_size=args.min_face_size,
        ),
    )
    if args.limit_samples is not None:
        dataset.records = dataset.records[: args.limit_samples]
    logger(f"train_dataset samples={len(dataset)} ann_file={args.ann_file} img_prefix={args.img_prefix}")

    data_loader = _build_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.workers > 0 and not args.no_persistent_workers,
        pin_memory=device.type == "cuda" and not args.no_pin_memory,
    )
    val_loader = _build_val_loader(args, device) if args.eval_interval > 0 else None
    logger(
        f"train_loader steps={len(data_loader)} batch_size={args.batch_size} "
        f"workers={args.workers} prefetch_factor={args.prefetch_factor} "
        f"persistent_workers={args.workers > 0 and not args.no_persistent_workers} "
        f"pin_memory={device.type == 'cuda' and not args.no_pin_memory}"
    )
    if val_loader is not None:
        logger(f"val_loader steps={len(val_loader)} eval_interval={args.eval_interval}")

    writer = _build_summary_writer(args.work_dir, disabled=args.no_tensorboard, logger=logger)
    train_started_at = time.perf_counter()
    remaining_epochs = max(args.epochs - start_epoch + 1, 0)
    total_train_steps = len(data_loader) * remaining_epochs
    try:
        if start_epoch > args.epochs:
            logger(f"nothing_to_train start_epoch={start_epoch} epochs={args.epochs}")
        for epoch in range(start_epoch, args.epochs + 1):
            completed_steps_before_epoch = (epoch - start_epoch) * len(data_loader)
            logger(
                f"start epoch={epoch}/{args.epochs} "
                f"steps={len(data_loader)} "
                f"batch_size={args.batch_size} "
                f"workers={args.workers} "
                f"prefetch_factor={args.prefetch_factor} "
                f"persistent_workers={args.workers > 0 and not args.no_persistent_workers} "
                f"pin_memory={device.type == 'cuda' and not args.no_pin_memory} "
                f"{_format_progress_eta(train_started_at, total_train_steps, completed_steps_before_epoch)}"
            )
            epoch_started_at = time.perf_counter()
            try:
                stats = train_one_epoch(
                    model=model,
                    criterion=criterion,
                    data_loader=data_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    lr_scheduler=lr_scheduler,
                    log_interval=args.log_interval,
                    logger=logger,
                    progress_suffix=lambda steps: _format_progress_eta(
                        train_started_at,
                        total_train_steps,
                        completed_steps_before_epoch + steps,
                    ),
                )
            except RuntimeError as exc:
                hinted_error = _with_dataloader_hint(exc, args)
                logger(f"error={hinted_error}")
                raise hinted_error from exc
            lr = optimizer.param_groups[0]["lr"]
            completed_train_steps = epoch * len(data_loader)
            progress_eta = _format_progress_eta(train_started_at, total_train_steps, completed_train_steps)
            eta_finish = _estimate_eta_finish(train_started_at, total_train_steps, completed_train_steps)
            elapsed_seconds = time.perf_counter() - train_started_at
            epoch_seconds = time.perf_counter() - epoch_started_at
            sec_per_step = epoch_seconds / max(stats.steps, 1)
            samples_per_second = stats.steps * args.batch_size / max(epoch_seconds, 1e-12)
            logger(
                f"epoch={epoch} "
                f"lr={lr:.8f} "
                f"loss={stats.loss:.6f} "
                f"cls={stats.loss_cls:.6f} "
                f"bbox={stats.loss_bbox:.6f} "
                f"obj={stats.loss_obj:.6f} "
                f"kps={stats.loss_kps:.6f} "
                f"epoch_seconds={epoch_seconds:.3f} "
                f"sec_per_step={sec_per_step:.4f} "
                f"samples_per_second={samples_per_second:.2f} "
                f"{progress_eta}"
            )
            _append_metrics_csv(
                args.work_dir / "metrics.csv",
                epoch,
                stats,
                lr=lr,
                elapsed_seconds=elapsed_seconds,
                eta_finish=eta_finish,
            )
            if writer is not None:
                _write_tensorboard(writer, epoch, stats, optimizer, prefix="train")
            latest_metrics = {
                "loss": stats.loss,
                "loss_cls": stats.loss_cls,
                "loss_bbox": stats.loss_bbox,
                "loss_obj": stats.loss_obj,
                "loss_kps": stats.loss_kps,
                "lr": lr,
                "best_loss": best_loss,
            }
            latest_path = args.work_dir / "latest.pth"
            _save_training_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics=latest_metrics,
                lr_scheduler=lr_scheduler,
            )
            logger(f"saved_latest_checkpoint path={latest_path}")
            if epoch % args.checkpoint_interval == 0:
                checkpoint_path = args.work_dir / f"epoch_{epoch}.pth"
                _save_training_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics=latest_metrics,
                    lr_scheduler=lr_scheduler,
                )
                logger(f"saved_checkpoint path={checkpoint_path}")
            if val_loader is not None and epoch % args.eval_interval == 0:
                logger(f"start_eval epoch={epoch} steps={len(val_loader)}")
                val_stats = evaluate_loss(
                    model=model,
                    criterion=criterion,
                    data_loader=val_loader,
                    device=device,
                )
                logger(
                    f"eval epoch={epoch} "
                    f"loss={val_stats.loss:.6f} "
                    f"cls={val_stats.loss_cls:.6f} "
                    f"bbox={val_stats.loss_bbox:.6f} "
                    f"obj={val_stats.loss_obj:.6f} "
                    f"kps={val_stats.loss_kps:.6f}"
                )
                _append_metrics_csv(
                    args.work_dir / "val_metrics.csv",
                    epoch,
                    val_stats,
                    lr=lr,
                    elapsed_seconds=time.perf_counter() - train_started_at,
                    eta_finish=_estimate_eta_finish(train_started_at, total_train_steps, completed_train_steps),
                )
                if writer is not None:
                    _write_tensorboard(writer, epoch, val_stats, optimizer, prefix="val")
                val_metrics = {
                    "val_loss": val_stats.loss,
                    "val_loss_cls": val_stats.loss_cls,
                    "val_loss_bbox": val_stats.loss_bbox,
                    "val_loss_obj": val_stats.loss_obj,
                    "val_loss_kps": val_stats.loss_kps,
                    "lr": lr,
                    "best_loss": best_loss,
                }
                eval_checkpoint_path = args.work_dir / f"eval_epoch_{epoch}.pth"
                _save_training_checkpoint(
                    path=eval_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                    lr_scheduler=lr_scheduler,
                )
                logger(f"saved_eval_checkpoint path={eval_checkpoint_path}")
                if best_loss is None or val_stats.loss < best_loss:
                    best_loss = val_stats.loss
                    best_metrics = {
                        **val_metrics,
                        "best_loss": best_loss,
                    }
                    best_path = args.work_dir / "best_loss.pth"
                    _save_training_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        metrics=best_metrics,
                        lr_scheduler=lr_scheduler,
                    )
                    _write_best_loss(args.work_dir, best_loss=best_loss, epoch=epoch)
                    logger(f"saved_best_checkpoint path={best_path} best_loss={best_loss:.6f}")
        logger(f"run_finished elapsed={_format_duration(time.perf_counter() - train_started_at)}")
    finally:
        if writer is not None:
            writer.close()
        logger.close()


def _build_val_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = WIDERFaceDataset(
        ann_file=args.val_ann_file,
        img_prefix=args.val_img_prefix,
        transform=build_eval_transforms(image_size=args.image_size),
        test_mode=True,
    )
    if args.eval_limit_samples is not None:
        dataset.records = dataset.records[: args.eval_limit_samples]

    return _build_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.workers > 0 and not args.no_persistent_workers,
        pin_memory=device.type == "cuda" and not args.no_pin_memory,
    )


def _build_data_loader(
    dataset: WIDERFaceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_face_samples,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
        kwargs["worker_init_fn"] = _init_worker
    return DataLoader(dataset, **kwargs)


def _init_worker(worker_id: int) -> None:
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _build_run_logger(args: argparse.Namespace) -> RunLogger:
    log_file = args.log_file or args.work_dir / "train.log"
    return RunLogger(log_file)


def _log_run_header(logger: RunLogger, args: argparse.Namespace) -> None:
    logger("=" * 80)
    logger(f"run_started_at={datetime.now():%Y-%m-%d %H:%M:%S}")
    logger(f"log_file={logger.path}")
    logger(f"python={sys.version.split()[0]} executable={sys.executable}")
    logger(f"platform={platform.platform()}")
    logger(
        f"torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda={torch.version.cuda}"
    )
    if torch.cuda.is_available():
        logger(f"cuda_device_count={torch.cuda.device_count()}")
        for device_idx in range(torch.cuda.device_count()):
            logger(f"cuda_device[{device_idx}]={torch.cuda.get_device_name(device_idx)}")
    for key, value in sorted(vars(args).items()):
        logger(f"arg.{key}={value}")


def _with_dataloader_hint(exc: RuntimeError, args: argparse.Namespace) -> RuntimeError:
    message = str(exc)
    if "DataLoader worker" not in message and "multiprocessing" not in message:
        return exc

    estimated_batch_mb = args.batch_size * 3 * args.image_size * args.image_size * 4 / 1024 / 1024
    queued_batches = args.workers * args.prefetch_factor if args.workers > 0 else 1
    hint = (
        f"{message}\n\n"
        "DataLoader worker crashed before an epoch finished. On Windows this often hides the real worker error. "
        "Retry once with `--workers 0` to expose the original exception. "
        f"Your current batch tensor is about {estimated_batch_mb:.1f} MB before labels/metas; "
        f"with workers={args.workers} and prefetch_factor={args.prefetch_factor}, "
        f"DataLoader may queue about {queued_batches} batches. "
        "For 640x640 training, prefer `--batch-size 16 --workers 4`, or lower memory pressure with "
        "`--batch-size 16`, `--prefetch-factor 1`, `--no-pin-memory`, or `--workers 0`."
    )
    return RuntimeError(hint)


def _save_training_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float | None],
    lr_scheduler: LinearWarmupMultiStepLR,
) -> None:
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        config=_serializable_config(args),
        metrics={key: value for key, value in metrics.items() if value is not None},
        scheduler_state=lr_scheduler.state_dict(),
    )


def _read_best_loss(work_dir: Path) -> float | None:
    best_file = work_dir / "best_loss.txt"
    if not best_file.exists():
        return None
    try:
        first_line = best_file.read_text(encoding="utf-8").splitlines()[0]
        return float(first_line.split(",", maxsplit=1)[0])
    except (IndexError, ValueError):
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


def _format_progress_eta(started_at: float, total_steps: int, completed_steps: int) -> str:
    if completed_steps <= 0:
        return "elapsed=00:00:00 eta=estimating"

    elapsed_seconds = time.perf_counter() - started_at
    if elapsed_seconds <= 0:
        return "elapsed=00:00:00 eta=estimating"

    steps_per_second = completed_steps / elapsed_seconds
    remaining_steps = max(total_steps - completed_steps, 0)
    if steps_per_second <= 0:
        return f"elapsed={_format_duration(elapsed_seconds)} eta=estimating"

    remaining_seconds = remaining_steps / steps_per_second
    eta_finish = datetime.now() + timedelta(seconds=remaining_seconds)
    return (
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"remaining={_format_duration(remaining_seconds)} "
        f"eta={eta_finish:%Y-%m-%d %H:%M:%S}"
    )


def _estimate_eta_finish(started_at: float, total_steps: int, completed_steps: int) -> str:
    if completed_steps <= 0:
        return ""
    elapsed_seconds = time.perf_counter() - started_at
    if elapsed_seconds <= 0:
        return ""
    steps_per_second = completed_steps / elapsed_seconds
    if steps_per_second <= 0:
        return ""
    remaining_seconds = max(total_steps - completed_steps, 0) / steps_per_second
    return f"{datetime.now() + timedelta(seconds=remaining_seconds):%Y-%m-%d %H:%M:%S}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _serializable_config(args: argparse.Namespace) -> dict[str, object]:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, torch.device):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _append_metrics_csv(
    path: Path,
    epoch: int,
    stats: Any,
    *,
    lr: float,
    elapsed_seconds: float | None = None,
    eta_finish: str = "",
) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "epoch",
                "lr",
                "loss",
                "loss_cls",
                "loss_bbox",
                "loss_obj",
                "loss_kps",
                "steps",
                "elapsed_seconds",
                "eta_finish",
            ),
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
                "loss_kps": stats.loss_kps,
                "steps": stats.steps,
                "elapsed_seconds": "" if elapsed_seconds is None else f"{elapsed_seconds:.3f}",
                "eta_finish": eta_finish,
            }
        )


def _build_summary_writer(work_dir: Path, *, disabled: bool, logger: RunLogger) -> Any | None:
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger("TensorBoard is not installed; skipping TensorBoard logs.")
        return None
    return SummaryWriter(log_dir=str(work_dir / "tensorboard"))


def _write_tensorboard(writer: Any, epoch: int, stats: Any, optimizer: torch.optim.Optimizer, *, prefix: str) -> None:
    writer.add_scalar(f"{prefix}/loss_total", stats.loss, epoch)
    writer.add_scalar(f"{prefix}/loss_cls", stats.loss_cls, epoch)
    writer.add_scalar(f"{prefix}/loss_bbox", stats.loss_bbox, epoch)
    writer.add_scalar(f"{prefix}/loss_obj", stats.loss_obj, epoch)
    writer.add_scalar(f"{prefix}/loss_kps", stats.loss_kps, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)


if __name__ == "__main__":
    main()
