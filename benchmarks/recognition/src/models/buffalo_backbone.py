import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, buffalo_model: BuffaloTorchModel, img_size: int = INPUT_SIZE):
        super().__init__()
        self.buffalo_model = buffalo_model
        self.img_size = img_size
        self.embedding_dim = buffalo_model.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return self.buffalo_model(x)
