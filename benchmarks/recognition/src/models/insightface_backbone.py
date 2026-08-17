"""Wraps the pretrained InsightFace `buffalo_l` face-recognition model
(ArcFace ResNet-50, w600k_r50.onnx, trained on real human faces from
WebFace600K) as a baseline to compare against before training our own model
on iCartoonFace/data-film.

Loaded via insightface.app.FaceAnalysis(name="buffalo_l"), which downloads
the whole model pack (detection + recognition + landmarks + age/gender) into
~/.insightface/models/buffalo_l/ on first use (FaceAnalysis requires a
'detection' model to be present even if unused, hence allowed_modules
includes it below) - only the 'recognition' sub-model (512-d ArcFace
embeddings) is actually run; face detection is never invoked, since we
already have this project's own bboxes and want to isolate the recognition
model's accuracy under the exact same protocol used for our own model.

Caveat: InsightFace's recognition models are normally evaluated on faces
aligned via 5-point facial landmarks (from their own detector), not a plain
bbox crop. We don't have landmarks for iCartoonFace/data-film, so this
baseline instead reuses the same bbox-crop-and-enlarge preprocessing as our
own model (src/datasets/icartoonface.py's crop_face()). This is an
approximation of buffalo_l's ideal input distribution, not a faithful
reproduction of it - read its numbers as "what this pretrained model scores
under our evaluation protocol", not as directly comparable to InsightFace's
own published benchmarks (LFW, etc.), which use proper alignment.
"""
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
        """x: (B, 3, H, W), normalized to roughly [-1, 1] by
        src/datasets/icartoonface.py's EVAL_TRANSFORM (mean=std=0.5). Resizes
        to InsightFace's expected 112x112 BGR input and returns raw (not
        L2-normalized) embeddings - src/evaluate.py's embed_paths()
        normalizes the output itself, same as for our own model."""
        device = x.device
        imgs = ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3) RGB

        bgr_batch = []
        for img in imgs:
            resized = Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            bgr_batch.append(np.array(resized)[:, :, ::-1])  # RGB -> BGR

        feats = self.rec_model.get_feat(bgr_batch)  # (B, 512) numpy, un-normalized
        return torch.from_numpy(feats).to(device)
