from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import init_yunet_weights
from .layers import Conv4layerBlock, Conv_head


class YuNetBackbone(nn.Module):
    def __init__(
        self,
        stage_channels: tuple[tuple[int, ...], ...],
        downsample_idx: tuple[int, ...],
        out_idx: tuple[int, ...],
    ):
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx
        self.model0 = Conv_head(*stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f"model{i}", Conv4layerBlock(*stage_channels[i]))
        self.init_weights()

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i in range(self.layer_num):
            x = getattr(self, f"model{i}")(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out

