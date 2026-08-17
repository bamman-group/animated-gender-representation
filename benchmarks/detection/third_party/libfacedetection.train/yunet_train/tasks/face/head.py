from __future__ import annotations

import torch
import torch.nn as nn

from yunet_train.models.init import init_yunet_weights
from yunet_train.models.layers import ConvDPUnit


class YuNetHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        shared_stacked_convs: int,
        stacked_convs: int,
        strides: tuple[int, ...],
        use_kps: bool = True,
        kps_num: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.NK = kps_num
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid_cls = True
        self.use_kps = use_kps
        self.shared_stack_convs = shared_stacked_convs
        self.strides = tuple((stride, stride) for stride in strides)
        self.strides_num = len(self.strides)

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
        if self.use_kps:
            self.multi_level_kps = nn.ModuleList()

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
                    chn = (
                        self.in_channels
                        if i == 0 and self.shared_stack_convs == 0
                        else self.feat_channels
                    )
                    single_level_cls_convs.append(ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_reg_convs.append(nn.Sequential(*single_level_reg_convs))
                self.multi_level_cls_convs.append(nn.Sequential(*single_level_cls_convs))

            chn = (
                self.in_channels
                if self.stacked_convs == 0 and self.shared_stack_convs == 0
                else self.feat_channels
            )
            self.multi_level_cls.append(ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            if self.use_kps:
                self.multi_level_kps.append(ConvDPUnit(chn, self.NK * 2, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(
        self,
        feats: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.shared_stack_convs > 0:
            feats = [
                convs(feat)
                for feat, convs in zip(feats, self.multi_level_share_convs)
            ]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [
                convs(feat)
                for feat, convs in zip(feats_cls, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_kps)
            ] if self.use_kps else []
        else:
            cls_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_kps)
            ] if self.use_kps else []

        if torch.onnx.is_in_onnx_export():
            cls = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, self.num_classes).sigmoid()
                for f in cls_preds
            ]
            obj = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, 1).sigmoid()
                for f in obj_preds
            ]
            bbox = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, 4)
                for f in bbox_preds
            ]
            kps = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, self.NK * 2)
                for f in kps_preds
            ]
            return cls, obj, bbox, kps

        return cls_preds, bbox_preds, obj_preds, kps_preds

