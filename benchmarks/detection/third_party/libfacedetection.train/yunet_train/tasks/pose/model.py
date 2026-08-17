from __future__ import annotations

import torch
import torch.nn as nn

from yunet_train.models.config import YuNetModelConfig, get_model_config
from yunet_train.models.backbone import YuNetBackbone
from yunet_train.models.init import init_yunet_weights
from yunet_train.models.layers import ConvDPUnit
from yunet_train.models.neck import TFPN


class YuNetPoseHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        shared_stacked_convs: int,
        stacked_convs: int,
        strides: tuple[int, ...],
        kpt_shape: tuple[int, int] = (17, 3),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.shared_stack_convs = shared_stacked_convs
        self.stacked_convs = stacked_convs
        self.strides = tuple((stride, stride) for stride in strides)
        self.strides_num = len(self.strides)
        self.kpt_shape = kpt_shape
        self.kpt_out_channels = kpt_shape[0] * kpt_shape[1]

        self._init_layers()
        self.init_weights()

    def _init_layers(self) -> None:
        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList()
        if self.stacked_convs > 0:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        self.multi_level_kpts = nn.ModuleList()

        for _ in self.strides:
            if self.shared_stack_convs > 0:
                single_level_share_convs = []
                for i in range(self.shared_stack_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    single_level_share_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_share_convs.append(nn.Sequential(*single_level_share_convs))

            if self.stacked_convs > 0:
                single_level_cls_convs = []
                single_level_reg_convs = []
                for i in range(self.stacked_convs):
                    chn = self.in_channels if i == 0 and self.shared_stack_convs == 0 else self.feat_channels
                    single_level_cls_convs.append(ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_cls_convs.append(nn.Sequential(*single_level_cls_convs))
                self.multi_level_reg_convs.append(nn.Sequential(*single_level_reg_convs))

            chn = self.in_channels if self.stacked_convs == 0 and self.shared_stack_convs == 0 else self.feat_channels
            kpt_channels = max(chn // 4, self.kpt_out_channels)
            self.multi_level_cls.append(ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))
            self.multi_level_kpts.append(
                nn.Sequential(
                    ConvDPUnit(chn, kpt_channels),
                    ConvDPUnit(kpt_channels, kpt_channels),
                    ConvDPUnit(kpt_channels, self.kpt_out_channels, False),
                )
            )

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(
        self,
        feats: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.shared_stack_convs > 0:
            feats = [convs(feat) for feat, convs in zip(feats, self.multi_level_share_convs)]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [convs(feat) for feat, convs in zip(feats_cls, self.multi_level_cls)]
            bbox_preds = [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_bbox)]
            obj_preds = [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_obj)]
            kpt_preds = [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_kpts)]
        else:
            cls_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_cls)]
            bbox_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_bbox)]
            obj_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_obj)]
            kpt_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_kpts)]

        if torch.onnx.is_in_onnx_export():
            cls = [
                feat.permute(0, 2, 3, 1).view(feat.shape[0], -1, self.num_classes).sigmoid()
                for feat in cls_preds
            ]
            obj = [feat.permute(0, 2, 3, 1).view(feat.shape[0], -1, 1).sigmoid() for feat in obj_preds]
            bbox = [feat.permute(0, 2, 3, 1).view(feat.shape[0], -1, 4) for feat in bbox_preds]
            kpts = [
                feat.permute(0, 2, 3, 1).view(feat.shape[0], -1, self.kpt_out_channels)
                for feat in kpt_preds
            ]
            return cls, obj, bbox, kpts

        return cls_preds, bbox_preds, obj_preds, kpt_preds


class YuNetPose(nn.Module):
    def __init__(
        self,
        config: YuNetModelConfig,
        *,
        num_classes: int = 1,
        kpt_shape: tuple[int, int] = (17, 3),
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.kpt_shape = kpt_shape
        self.backbone = YuNetBackbone(
            stage_channels=config.stage_channels,
            downsample_idx=config.downsample_idx,
            out_idx=config.out_idx,
        )
        self.neck = TFPN(
            in_channels=config.neck_in_channels,
            out_idx=config.neck_out_idx,
        )
        self.pose_head = YuNetPoseHead(
            num_classes=num_classes,
            in_channels=config.in_channels,
            feat_channels=config.feat_channels,
            shared_stacked_convs=config.shared_stacked_convs,
            stacked_convs=config.stacked_convs,
            strides=config.strides,
            kpt_shape=kpt_shape,
        )

    def extract_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(img)
        return self.neck(feats)

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        feats = self.extract_feat(img)
        return self.pose_head(feats)


def build_yunet_pose(
    variant: str = "yunet_n",
    *,
    num_classes: int = 1,
    kpt_shape: tuple[int, int] = (17, 3),
) -> YuNetPose:
    return YuNetPose(get_model_config(variant), num_classes=num_classes, kpt_shape=kpt_shape)
