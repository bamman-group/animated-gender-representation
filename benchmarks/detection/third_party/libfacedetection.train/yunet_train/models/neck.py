from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import init_yunet_weights
from .layers import ConvDPUnit


class TFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], out_idx: tuple[int, ...]):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(ConvDPUnit(in_channels[i], in_channels[i], True))
        self.init_weights()

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(feats)
        num_feats = len(feats)

        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i],
                scale_factor=2.0,
                mode="nearest",
            )

        feats[0] = self.lateral_convs[0](feats[0])
        return [feats[i] for i in self.out_idx]

