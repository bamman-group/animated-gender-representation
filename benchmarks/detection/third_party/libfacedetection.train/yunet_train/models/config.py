from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YuNetModelConfig:
    variant: str
    stage_channels: tuple[tuple[int, ...], ...]
    downsample_idx: tuple[int, ...]
    out_idx: tuple[int, ...]
    neck_in_channels: tuple[int, ...]
    neck_out_idx: tuple[int, ...]
    num_classes: int
    in_channels: int
    feat_channels: int
    shared_stacked_convs: int
    stacked_convs: int
    strides: tuple[int, ...]
    use_kps: bool
    kps_num: int


YUNET_N = YuNetModelConfig(
    variant="yunet_n",
    stage_channels=(
        (3, 16, 16),
        (16, 64),
        (64, 64),
        (64, 64),
        (64, 64),
        (64, 64),
    ),
    downsample_idx=(0, 2, 3, 4),
    out_idx=(3, 4, 5),
    neck_in_channels=(64, 64, 64),
    neck_out_idx=(0, 1, 2),
    num_classes=1,
    in_channels=64,
    feat_channels=64,
    shared_stacked_convs=1,
    stacked_convs=0,
    strides=(8, 16, 32),
    use_kps=True,
    kps_num=5,
)

YUNET_S = YuNetModelConfig(
    variant="yunet_s",
    stage_channels=(
        (3, 16, 16),
        (16, 32),
        (32, 64),
        (64, 64),
        (64, 64),
        (64, 64),
    ),
    downsample_idx=(0, 2, 3, 4),
    out_idx=(3, 4, 5),
    neck_in_channels=(64, 64, 64),
    neck_out_idx=(0, 1, 2),
    num_classes=1,
    in_channels=64,
    feat_channels=64,
    shared_stacked_convs=0,
    stacked_convs=0,
    strides=(8, 16, 32),
    use_kps=True,
    kps_num=5,
)

MODEL_CONFIGS = {
    YUNET_N.variant: YUNET_N,
    YUNET_S.variant: YUNET_S,
}


def get_model_config(variant: str) -> YuNetModelConfig:
    try:
        return MODEL_CONFIGS[variant]
    except KeyError as exc:
        names = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown YuNet variant {variant!r}. Expected one of: {names}") from exc
