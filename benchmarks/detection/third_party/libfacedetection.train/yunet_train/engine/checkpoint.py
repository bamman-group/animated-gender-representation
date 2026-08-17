from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_model_weights_only(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model weights from a training checkpoint without optimizer or scheduler state."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(_clean_state_dict(state_dict), strict=True)
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(_clean_state_dict(state_dict), strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "lr_scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return checkpoint


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    scheduler_state: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if scheduler_state is not None:
        checkpoint["lr_scheduler"] = scheduler_state
    torch.save(checkpoint, checkpoint_path)


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned
