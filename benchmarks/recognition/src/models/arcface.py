"""Additive Angular Margin (ArcFace) classification head, used by
src/train_dino.py / src/train_buffalo.py when fine-tuning on the
iCartoonFace rectrain *identity* labels (--data-source identity --loss
arcface, the default).

Why ArcFace instead of the triplet loss the original external scripts used:
buffalo_l's own recognition backbone (w600k_r50) was trained with a
margin-softmax objective, and with clean identity labels a margin-softmax
classifier over the 5,013 training identities gives a much stronger, denser
gradient than sampling one random negative per anchor (where, with thousands
of identities, the negative is almost always already far away and the
triplet contributes no gradient - the collapsing-`active`-fraction problem).
The --data-source tracks path has no global identity labels, so it keeps the
triplet loss with its original co-occurring-face negatives (deliberately no
cross-frame negative mining: two tracks in different frames may be the same,
unlabeled person - see src/train_dino.py's TripletTrackDataset).

This head is a *training-time only* classifier: Rank@1 evaluation always uses
the backbone's L2-normalized embedding directly (via the eval adapters), so
the head is not part of the saved backbone checkpoint and does not need to be
loaded for evaluation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Maps an L2-normalized embedding (dim = in_features) to scaled,
    angular-margin logits suitable for nn.CrossEntropyLoss.

    Expects the incoming embeddings to already be unit-norm (every backbone
    wrapper in this repo L2-normalizes its output), so `embeddings @ W.T`
    with a normalized W is exactly the cosine similarity to each class
    prototype."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Beyond theta = pi - margin, cos(theta + m) stops being monotonic;
        # fall back to a linear penalty there (the standard ArcFace guard).
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        cosine = embeddings @ weight.t()  # (B, num_classes)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + margin)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
