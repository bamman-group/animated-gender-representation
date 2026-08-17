from __future__ import annotations

from pathlib import Path

import torch

from yunet_train.tasks.face import build_yunet


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def test_yunet_n_loads_legacy_checkpoint_without_key_mismatch() -> None:
    model = build_yunet("yunet_n")
    state_dict = _load_checkpoint(REPO_ROOT / "weights" / "yunet_n.pth")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    assert missing == []
    assert unexpected == []


def test_yunet_s_loads_legacy_checkpoint_without_key_mismatch() -> None:
    model = build_yunet("yunet_s")
    state_dict = _load_checkpoint(REPO_ROOT / "weights" / "yunet_s.pth")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    assert missing == []
    assert unexpected == []
