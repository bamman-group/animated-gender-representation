from __future__ import annotations

import torch
import torch.nn as nn

from yunet_train.models.config import YuNetModelConfig, get_model_config

from yunet_train.models.backbone import YuNetBackbone
from yunet_train.models.neck import TFPN

from .head import YuNetHead


class YuNet(nn.Module):
    def __init__(self, config: YuNetModelConfig):
        super().__init__()
        self.config = config
        self.backbone = YuNetBackbone(
            stage_channels=config.stage_channels,
            downsample_idx=config.downsample_idx,
            out_idx=config.out_idx,
        )
        self.neck = TFPN(
            in_channels=config.neck_in_channels,
            out_idx=config.neck_out_idx,
        )
        self.bbox_head = YuNetHead(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            feat_channels=config.feat_channels,
            shared_stacked_convs=config.shared_stacked_convs,
            stacked_convs=config.stacked_convs,
            strides=config.strides,
            use_kps=config.use_kps,
            kps_num=config.kps_num,
        )

    def extract_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(img)
        return self.neck(feats)

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        feats = self.extract_feat(img)
        return self.bbox_head(feats)


def build_yunet(variant: str = "yunet_n") -> YuNet:
    return YuNet(get_model_config(variant))

