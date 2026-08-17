"""YOLO-style mosaic augmentation for RetinaFace training.

Wraps a detection dataset that yields raw (BGR image, Nx15 target in pixel
coords: 4 box corners, 10 landmark coords, landmark flag). With probability
`prob`, four images are stitched onto a 2s x 2s canvas around a random
center and the combined target is remapped; the standard preproc (random
crop, distort, flip, resize to s x s) then runs on the composite. Otherwise
the single image goes through preproc as usual.

Boxes clipped by the canvas are kept if enough of them survives; landmarks
are kept only if the box survives intact enough and all points stay inside
the canvas, otherwise the face's landmark flag is set to -1 (loss masked).
"""
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data


class MosaicDetection(data.Dataset):
    def __init__(self, dataset, preproc, img_dim, prob=0.5,
                 min_box_px=2.0, min_visibility=0.3):
        """dataset must be constructed with preproc=None (raw samples)."""
        self.dataset = dataset
        self.preproc = preproc
        self.img_dim = img_dim
        self.prob = prob
        self.min_box_px = min_box_px
        self.min_visibility = min_visibility

    def __len__(self):
        return len(self.dataset)

    def _raw(self, index):
        img, target = self.dataset[index]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        return img, np.asarray(target, dtype=np.float64).reshape(-1, 15)

    def _mosaic(self, index):
        s = self.img_dim
        indices = [index] + [random.randrange(len(self.dataset)) for _ in range(3)]
        canvas = np.full((2 * s, 2 * s, 3), 114.0, dtype=np.float32)
        yc = random.randint(s // 2, 3 * s // 2)
        xc = random.randint(s // 2, 3 * s // 2)

        targets = []
        for i, idx in enumerate(indices):
            img, target = self._raw(idx)
            h, w = img.shape[:2]
            scale = s / max(h, w)
            if scale != 1.0:
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]

            # quadrant placement (canvas region and matching image region)
            if i == 0:    # top-left
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x2a, y2a = xc, yc
            elif i == 1:  # top-right
                x1a, y1a = xc, max(yc - h, 0)
                x2a, y2a = min(xc + w, 2 * s), yc
            elif i == 2:  # bottom-left
                x1a, y1a = max(xc - w, 0), yc
                x2a, y2a = xc, min(yc + h, 2 * s)
            else:         # bottom-right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + w, 2 * s), min(yc + h, 2 * s)

            # image crop pasted into that region (anchored at the center corner)
            if i == 0:
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif i == 1:
                x1b, y1b = 0, h - (y2a - y1a)
            elif i == 2:
                x1b, y1b = w - (x2a - x1a), 0
            else:
                x1b, y1b = 0, 0
            x2b, y2b = x1b + (x2a - x1a), y1b + (y2a - y1a)
            if x2a <= x1a or y2a <= y1a:
                continue
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            pad_x, pad_y = x1a - x1b, y1a - y1b
            t = target.copy()
            t[:, [0, 2]] = t[:, [0, 2]] * scale + pad_x
            t[:, [1, 3]] = t[:, [1, 3]] * scale + pad_y
            t[:, 4:14:2] = np.where(t[:, 4:14:2] >= 0,
                                    t[:, 4:14:2] * scale + pad_x, -1.0)
            t[:, 5:14:2] = np.where(t[:, 5:14:2] >= 0,
                                    t[:, 5:14:2] * scale + pad_y, -1.0)

            # clip boxes to the pasted region; keep sufficiently visible ones
            area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
            t[:, 0] = t[:, 0].clip(x1a, x2a)
            t[:, 1] = t[:, 1].clip(y1a, y2a)
            t[:, 2] = t[:, 2].clip(x1a, x2a)
            t[:, 3] = t[:, 3].clip(y1a, y2a)
            w_c = t[:, 2] - t[:, 0]
            h_c = t[:, 3] - t[:, 1]
            vis = np.where(area > 0, w_c * h_c / np.maximum(area, 1e-9), 0)
            keep = ((w_c >= self.min_box_px) & (h_c >= self.min_box_px)
                    & (vis >= self.min_visibility))
            t = t[keep]
            if not len(t):
                continue

            # landmarks survive only if the box wasn't meaningfully clipped
            # and every point lies inside the pasted region
            lm_ok = vis[keep] >= 0.99
            for j in range(len(t)):
                pts_x = t[j, 4:14:2]
                pts_y = t[j, 5:14:2]
                inside = ((pts_x >= x1a) & (pts_x < x2a)
                          & (pts_y >= y1a) & (pts_y < y2a)).all()
                if t[j, 14] != 1 or not lm_ok[j] or not inside:
                    t[j, 4:14] = -1.0
                    t[j, 14] = -1.0
            targets.append(t)

        if not targets:
            return None
        return canvas, np.concatenate(targets, axis=0)

    def __getitem__(self, index):
        if random.random() < self.prob:
            result = self._mosaic(index)
            if result is not None:
                img, target = result
                img, target = self.preproc(img, target)
                return torch.from_numpy(img), target
        img, target = self._raw(index)
        img, target = self.preproc(np.float32(img), target)
        return torch.from_numpy(img), target
