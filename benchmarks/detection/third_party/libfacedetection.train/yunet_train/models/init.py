from __future__ import annotations

import torch.nn as nn


def init_yunet_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            if layer.bias is not None:
                nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.fill_(0.02)
            else:
                layer.weight.data.normal_(0, 0.01)
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()

