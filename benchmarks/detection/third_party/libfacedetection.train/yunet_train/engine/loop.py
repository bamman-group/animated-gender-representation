from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader


class LRScheduler(Protocol):
    def step(self, *, epoch: int) -> list[float]:
        ...


def train_loss_epoch(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    move_batch: Callable[[Any, torch.device | str], Any],
    compute_losses: Callable[[torch.nn.Module, Any], dict[str, torch.Tensor]],
    loss_names: tuple[str, ...],
    device: torch.device | str,
    epoch: int = 1,
    lr_scheduler: LRScheduler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    logger: Callable[[str], None] | None = None,
    format_log: Callable[[int, int, int, dict[str, float]], str] | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = _empty_totals(loss_names)
    steps = 0

    for batch in data_loader:
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, batch)
        loss = sum(losses.values())
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        steps += 1
        _accumulate(totals, losses, loss, loss_names)
        if log_interval > 0 and logger is not None and format_log is not None and (steps == 1 or steps % log_interval == 0):
            logger(format_log(epoch, steps, len(data_loader), totals))

    if steps == 0:
        raise ValueError("data_loader yielded no batches")
    return totals, steps


@torch.no_grad()
def evaluate_loss_epoch(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    move_batch: Callable[[Any, torch.device | str], Any],
    compute_losses: Callable[[torch.nn.Module, Any], dict[str, torch.Tensor]],
    loss_names: tuple[str, ...],
    device: torch.device | str,
) -> tuple[dict[str, float], int]:
    model.eval()
    totals = _empty_totals(loss_names)
    steps = 0
    for batch in data_loader:
        batch = move_batch(batch, device)
        losses = compute_losses(model, batch)
        loss = sum(losses.values())
        steps += 1
        _accumulate(totals, losses, loss, loss_names)

    if steps == 0:
        raise ValueError("data_loader yielded no batches")
    return totals, steps


def _empty_totals(loss_names: tuple[str, ...]) -> dict[str, float]:
    totals = {"loss": 0.0}
    totals.update({name: 0.0 for name in loss_names})
    return totals


def _accumulate(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    loss: torch.Tensor,
    loss_names: tuple[str, ...],
) -> None:
    totals["loss"] += float(loss.detach().cpu())
    for name in loss_names:
        totals[name] += float(losses[name].detach().cpu())
