"""Mosaic augmentation for YuNet training WITHOUT forking the vendored
libfacedetection.train code.

MosaicWIDERFaceDataset subclasses the upstream WIDERFaceDataset: with
probability `prob` it composes four raw samples onto a 2s x 2s canvas
(remapping boxes and 5-point keypoints, dropping faces clipped below 30%
visibility, and un-annotating keypoints of clipped faces), then applies the
upstream training transform (RandomSquareCrop / Resize / flip / ...). The
close-mosaic wind-down is approximated inside workers by counting served
samples.

train_yunet.py injects this class into yunet_train.cli.train by name, so the
vendored upstream code itself is unmodified.
"""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "libfacedetection.train"))

from yunet_train.tasks.face import WIDERFaceDataset  # noqa: E402


class MosaicWIDERFaceDataset(WIDERFaceDataset):
    # class-level knobs, set once by train_yunet.py before dataset creation
    # (the upstream trainer constructs the dataset itself, so constructor
    # signature must stay upstream-compatible)
    MOSAIC_PROB = 1.0
    IMAGE_SIZE = 640
    TOTAL_EPOCHS = None      # for the close-mosaic approximation
    CLOSE_EPOCHS = 10
    WORKERS = 1
    MIN_VISIBILITY = 0.3

    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, transform=None, **kwargs)
        self._final_transform = transform
        self._served = 0

    def _mosaic_active(self):
        if self.MOSAIC_PROB <= 0 or self.test_mode:
            return False
        if self.TOTAL_EPOCHS:
            # per-worker epoch estimate: each worker serves ~len/WORKERS
            # samples per epoch (persistent workers)
            per_epoch = max(1, len(self) // max(self.WORKERS, 1))
            epoch = self._served / per_epoch
            if epoch >= self.TOTAL_EPOCHS - self.CLOSE_EPOCHS:
                return False
        return random.random() < self.MOSAIC_PROB

    def _raw(self, index):
        record = self.records[index]
        return self._record_to_sample(record)

    def __getitem__(self, index):
        self._served += 1
        if self._mosaic_active():
            sample = self._compose_mosaic(index)
        else:
            sample = self._raw(index)
        if self._final_transform is not None:
            sample = self._final_transform(sample)
        return sample

    def _compose_mosaic(self, index):
        s = self.IMAGE_SIZE
        indices = [index] + [random.randrange(len(self)) for _ in range(3)]
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        yc = random.randint(s // 2, 3 * s // 2)
        xc = random.randint(s // 2, 3 * s // 2)

        boxes, labels, keypoints = [], [], []
        ig_boxes, ig_labels = [], []
        base = None
        for i, idx in enumerate(indices):
            sample = self._raw(idx)
            if base is None:
                base = sample
            img = np.asarray(sample.image)
            h, w = img.shape[:2]
            scale = s / max(h, w)
            if scale != 1.0:
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
            if x2a <= x1a or y2a <= y1a:
                continue
            x1b = w - (x2a - x1a) if i in (0, 2) else 0
            y1b = h - (y2a - y1a) if i in (0, 1) else 0
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a),
                                           x1b:x1b + (x2a - x1a)]
            pad_x, pad_y = x1a - x1b, y1a - y1b

            def remap_boxes(b):
                b = np.asarray(b, dtype=np.float32).reshape(-1, 4).copy()
                b *= scale
                b[:, [0, 2]] += pad_x
                b[:, [1, 3]] += pad_y
                return b

            b = remap_boxes(sample.boxes)
            kp = np.asarray(sample.keypoints, dtype=np.float32).copy()
            if kp.size:
                kp = kp.reshape(len(b), -1, 3)
                ann = kp[:, :, 2] >= 0
                kp[:, :, 0] = np.where(ann, kp[:, :, 0] * scale + pad_x, -1)
                kp[:, :, 1] = np.where(ann, kp[:, :, 1] * scale + pad_y, -1)

            # clip to pasted region; keep sufficiently visible faces
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b[:, [0, 2]] = b[:, [0, 2]].clip(x1a, x2a)
            b[:, [1, 3]] = b[:, [1, 3]].clip(y1a, y2a)
            w_c, h_c = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            vis = np.where(area > 0, w_c * h_c / np.maximum(area, 1e-9), 0)
            keep = (w_c >= 2) & (h_c >= 2) & (vis >= self.MIN_VISIBILITY)
            if keep.any():
                bk = b[keep]
                kpk = kp[keep] if kp.size else kp.reshape(0, 0, 3)
                if kpk.size:
                    # un-annotate keypoints of clipped faces or points that
                    # left the pasted region
                    for j in range(len(bk)):
                        pts = kpk[j]
                        inside = ((pts[:, 0] >= x1a) & (pts[:, 0] < x2a)
                                  & (pts[:, 1] >= y1a) & (pts[:, 1] < y2a))
                        if vis[keep][j] < 0.99 or not (inside | (pts[:, 2] < 0)).all():
                            kpk[j] = -1.0
                boxes.append(bk)
                labels.append(np.asarray(sample.labels)[keep])
                keypoints.append(kpk)

            igb = np.asarray(sample.ignored_boxes, dtype=np.float32).reshape(-1, 4)
            if len(igb):
                igb = remap_boxes(igb)
                igb[:, [0, 2]] = igb[:, [0, 2]].clip(x1a, x2a)
                igb[:, [1, 3]] = igb[:, [1, 3]].clip(y1a, y2a)
                ok = ((igb[:, 2] - igb[:, 0]) >= 2) & ((igb[:, 3] - igb[:, 1]) >= 2)
                if ok.any():
                    ig_boxes.append(igb[ok])
                    ig_labels.append(np.asarray(sample.ignored_labels)[ok])

        if not boxes:                    # degenerate composition: fall back
            return self._raw(index)

        kp_dim = max(k.shape[1] for k in keypoints if k.ndim == 3)
        return replace(
            base,
            image=canvas,
            boxes=np.concatenate(boxes, axis=0),
            labels=np.concatenate(labels, axis=0),
            keypoints=np.concatenate(
                [k.reshape(-1, kp_dim, 3) for k in keypoints], axis=0),
            ignored_boxes=(np.concatenate(ig_boxes, axis=0) if ig_boxes
                           else np.zeros((0, 4), dtype=np.float32)),
            ignored_labels=(np.concatenate(ig_labels, axis=0) if ig_labels
                            else np.zeros((0,), dtype=np.int64)),
            original_shape=canvas.shape,
            image_shape=canvas.shape,
        )
