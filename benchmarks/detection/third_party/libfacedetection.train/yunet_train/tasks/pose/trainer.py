from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader

from yunet_train.engine.loop import LRScheduler, evaluate_loss_epoch, train_loss_epoch

from .types import PoseBatch


class PoseCriterion(Protocol):
    def __call__(
        self,
        preds: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        *,
        boxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        keypoints: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...


LOSS_NAMES = ("loss_cls", "loss_bbox", "loss_obj", "loss_kpt", "loss_kpt_vis")


@dataclass(frozen=True)
class PoseTrainStats:
    loss: float
    loss_cls: float
    loss_bbox: float
    loss_obj: float
    loss_kpt: float
    loss_kpt_vis: float
    steps: int


def move_pose_batch_to_device(batch: PoseBatch, device: torch.device | str) -> PoseBatch:
    return PoseBatch(
        images=batch.images.to(device, non_blocking=True),
        boxes=[boxes.to(device, non_blocking=True) for boxes in batch.boxes],
        labels=[labels.to(device, non_blocking=True) for labels in batch.labels],
        keypoints=[keypoints.to(device, non_blocking=True) for keypoints in batch.keypoints],
        metas=batch.metas,
    )


def train_pose_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: PoseCriterion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int = 1,
    lr_scheduler: LRScheduler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    logger: Callable[[str], None] | None = None,
) -> PoseTrainStats:
    totals, steps = train_loss_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        move_batch=move_pose_batch_to_device,
        compute_losses=lambda model, batch: _compute_losses(model, criterion, batch),
        loss_names=LOSS_NAMES,
        device=device,
        epoch=epoch,
        lr_scheduler=lr_scheduler,
        grad_clip_norm=grad_clip_norm,
        log_interval=log_interval,
        logger=logger,
        format_log=_format_step,
    )
    return _stats_from_totals(totals, steps)


def evaluate_pose_loss(
    *,
    model: torch.nn.Module,
    criterion: PoseCriterion,
    data_loader: DataLoader,
    device: torch.device | str,
) -> PoseTrainStats:
    totals, steps = evaluate_loss_epoch(
        model=model,
        data_loader=data_loader,
        move_batch=move_pose_batch_to_device,
        compute_losses=lambda model, batch: _compute_losses(model, criterion, batch),
        loss_names=LOSS_NAMES,
        device=device,
    )
    return _stats_from_totals(totals, steps)


def _compute_losses(model: torch.nn.Module, criterion: PoseCriterion, batch: PoseBatch) -> dict[str, torch.Tensor]:
    return criterion(
        model(batch.images),
        boxes=batch.boxes,
        labels=batch.labels,
        keypoints=batch.keypoints,
    )


def _stats_from_totals(totals: dict[str, float], steps: int) -> PoseTrainStats:
    return PoseTrainStats(
        loss=totals["loss"] / steps,
        loss_cls=totals["loss_cls"] / steps,
        loss_bbox=totals["loss_bbox"] / steps,
        loss_obj=totals["loss_obj"] / steps,
        loss_kpt=totals["loss_kpt"] / steps,
        loss_kpt_vis=totals["loss_kpt_vis"] / steps,
        steps=steps,
    )


def _format_step(epoch: int, steps: int, total_steps: int, totals: dict[str, float]) -> str:
    return (
        f"train epoch={epoch} "
        f"step={steps}/{total_steps} "
        f"loss={totals['loss'] / steps:.6f} "
        f"cls={totals['loss_cls'] / steps:.6f} "
        f"bbox={totals['loss_bbox'] / steps:.6f} "
        f"obj={totals['loss_obj'] / steps:.6f} "
        f"kpt={totals['loss_kpt'] / steps:.6f} "
        f"kpt_vis={totals['loss_kpt_vis'] / steps:.6f}"
    )
