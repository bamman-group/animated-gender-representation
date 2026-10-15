import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_EMBED_DIM = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoFaceModel(nn.Module):

    def __init__(self, backbone: nn.Module, embed_dim: int, proj_dim: int):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = proj_dim if proj_dim > 0 else embed_dim
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim),
            )
        else:
            self.proj = None

    def forward(self, x):
        feats = self.backbone(x)
        if self.proj is not None:
            feats = self.proj(feats)
        return F.normalize(feats, dim=1)


class DinoEvalAdapter(nn.Module):

    def __init__(self, dino_model: DinoFaceModel, img_size: int):
        super().__init__()
        self.dino_model = dino_model
        self.img_size = img_size
        self.embedding_dim = dino_model.embedding_dim
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1, 1) * 0.5 + 0.5
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.dino_model(x)


def _import_hubconf(dinov2_dir: str):
    if dinov2_dir not in sys.path:
        sys.path.insert(0, dinov2_dir)
    import hubconf

    return hubconf


def load_model(dinov2_dir: str, arch: str, weights_path: str, proj_dim: int) -> DinoFaceModel:
    hubconf = _import_hubconf(dinov2_dir)
    builder = getattr(hubconf, f"dinov2_{arch}")
    backbone = builder(pretrained=True, weights=weights_path)
    embed_dim = ARCH_EMBED_DIM[arch]
    return DinoFaceModel(backbone, embed_dim, proj_dim)


def load_model_from_checkpoint(checkpoint_path: str, dinov2_dir: str):
    import torch

    state = torch.load(checkpoint_path, map_location="cpu")
    arch = state["arch"]
    proj_dim = state["proj_dim"]
    embed_dim = ARCH_EMBED_DIM[arch]

    hubconf = _import_hubconf(dinov2_dir)
    builder = getattr(hubconf, f"dinov2_{arch}")
    backbone = builder(pretrained=False)
    model = DinoFaceModel(backbone, embed_dim, proj_dim)
    model.load_state_dict(state["model"])
    return model
