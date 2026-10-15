import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):

    def __init__(self, in_features: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        cosine = embeddings @ weight.t()
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
