from __future__ import annotations

from yunet_train.models.config import MODEL_CONFIGS, YUNET_N, YUNET_S, YuNetModelConfig, get_model_config

TRAIN_CROP_CHOICES = {
    "yunet_n": (0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    "yunet_s": (0.3, 0.45, 0.6, 0.8, 1.0),
}


def get_train_crop_choice(variant: str) -> tuple[float, ...]:
    try:
        return TRAIN_CROP_CHOICES[variant]
    except KeyError as exc:
        names = ", ".join(sorted(TRAIN_CROP_CHOICES))
        raise ValueError(f"Unknown YuNet variant {variant!r}. Expected one of: {names}") from exc


__all__ = [
    "MODEL_CONFIGS",
    "TRAIN_CROP_CHOICES",
    "YUNET_N",
    "YUNET_S",
    "YuNetModelConfig",
    "get_model_config",
    "get_train_crop_choice",
]
