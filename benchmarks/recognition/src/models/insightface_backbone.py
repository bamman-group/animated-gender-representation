import numpy as np
import torch
import torch.nn as nn
from PIL import Image

INPUT_SIZE = 112


class InsightFaceBuffaloL(nn.Module):
    def __init__(self, ctx_id: int = -1):
        super().__init__()
        from insightface.app import FaceAnalysis

        self.embedding_dim = 512
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=ctx_id)
        self.rec_model = app.models["recognition"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        imgs = ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

        bgr_batch = []
        for img in imgs:
            resized = Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            bgr_batch.append(np.array(resized)[:, :, ::-1])

        feats = self.rec_model.get_feat(bgr_batch)
        return torch.from_numpy(feats).to(device)
