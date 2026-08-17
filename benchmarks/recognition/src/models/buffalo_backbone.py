"""Wraps InsightFace buffalo_l's w600k_r50.onnx recognition backbone,
converted to a trainable torch.nn.Module via onnx2torch, for fine-tuning on
iCartoonFace identities - used by src/train_buffalo.py.

Unlike src/models/insightface_backbone.py::InsightFaceBuffaloL (which runs
the frozen ONNX graph directly through onnxruntime for inference-only
baseline evaluation, not differentiable/trainable), this module converts the
ONNX graph into actual PyTorch modules with trainable parameters so it can
be fine-tuned with backprop.

Requires the pip packages `onnx` and `onnx2torch` (see environment.yml).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# buffalo_l's w600k_r50.onnx native input size and normalization
# (mean=std=0.5 - same convention as src/datasets/icartoonface.py's own
# EVAL_TRANSFORM, unlike DINOv2's ImageNet stats).
INPUT_SIZE = 112
EMBEDDING_DIM = 512


def load_backbone_from_onnx(onnx_path: str) -> nn.Module:
    try:
        import onnx
        import onnx2torch
    except ImportError:
        raise ImportError("pip install onnx2torch onnx")
    print(f"Converting {onnx_path} -> PyTorch ...")
    backbone = onnx2torch.convert(onnx.load(onnx_path))
    backbone.eval()
    return backbone


def export_onnx(backbone: nn.Module, out_path: str) -> None:
    try:
        import onnx
    except ImportError:
        print("[warning] onnx not installed - skipping export.")
        return
    backbone = backbone.cpu().eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        backbone,
        dummy,
        str(out_path),
        input_names=["data"],
        output_names=["fc1"],
        dynamic_axes={"data": {0: "None"}, "fc1": {0: "None"}},
        opset_version=11,
        keep_initializers_as_inputs=False,
        verbose=False,
    )
    onnx.save(onnx.load(str(out_path)), str(out_path))
    print(f"  ONNX exported -> {out_path}")


class BuffaloTorchModel(nn.Module):
    """Trainable wrapper around the onnx2torch-converted buffalo_l backbone:
    handles the tuple/list output some onnx2torch conversions produce and
    L2-normalizes the embedding, matching the convention every other model
    in this repo follows (DinoFaceModel, InsightFaceBuffaloL, ...)."""

    def __init__(self, backbone: nn.Module, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return F.normalize(feats, dim=1)


class BuffaloEvalAdapter(nn.Module):
    """Wraps a BuffaloTorchModel so it can plug into src/evaluate.py's
    embed_paths() / src/evaluate_film.py's rank1_identification_accuracy(),
    exactly like src/models/dino_backbone.py's DinoEvalAdapter.

    Input: (B, 3, H, W) normalized to roughly [-1, 1] by
    src/datasets/icartoonface.py's EVAL_TRANSFORM (mean=std=0.5) - the same
    normalization convention buffalo_l itself uses, so only a resize to
    INPUT_SIZE (112) is needed here, no renormalization (unlike DINOv2's
    adapter, which also has to convert to ImageNet stats)."""

    def __init__(self, buffalo_model: BuffaloTorchModel, img_size: int = INPUT_SIZE):
        super().__init__()
        self.buffalo_model = buffalo_model
        self.img_size = img_size
        self.embedding_dim = buffalo_model.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return self.buffalo_model(x)
