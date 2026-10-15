"""Augmentations closing the gap to the ultralytics detect pipeline:

- zoom-out ("expand"): paste the image at a random position on a larger
  mean-colored canvas (ratio 1..expand_max) before the standard preproc crop,
  so faces also appear SMALLER than native (the stock RetinaFace crop only
  zooms in). Random placement doubles as free translation.
- low-probability photometric extras: blur, median blur, grayscale, CLAHE
  (each p=extras_prob, mirroring ultralytics' albumentations transforms).

Use ExtendedPreproc as a drop-in replacement for data.preproc.
"""
import random

import cv2
import numpy as np

from data.data_augment import preproc


def _photometric_extras(img, prob):
    """img: HxWx3 BGR (uint8 or float). Each op applies with probability prob."""
    if prob <= 0:
        return img
    as_float = img.dtype != np.uint8
    u8 = np.clip(img, 0, 255).astype(np.uint8) if as_float else img

    if random.random() < prob:
        u8 = cv2.blur(u8, (5, 5))
    if random.random() < prob:
        u8 = cv2.medianBlur(u8, 5)
    if random.random() < prob:
        gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
        u8 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if random.random() < prob:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return u8.astype(np.float32) if as_float else u8


def _expand(img, target, rgb_means, max_ratio):
    """Zoom-out: place the image on a canvas of ratio in (1, max_ratio]."""
    h, w = img.shape[:2]
    ratio = random.uniform(1.0, max_ratio)
    if ratio <= 1.001:
        return img, target
    ch, cw = int(h * ratio), int(w * ratio)
    canvas = np.empty((ch, cw, 3), dtype=np.float32)
    canvas[:, :] = rgb_means
    top = random.randint(0, ch - h)
    left = random.randint(0, cw - w)
    canvas[top:top + h, left:left + w] = img

    t = target.copy()
    t[:, [0, 2]] += left
    t[:, [1, 3]] += top
    # landmark coords: shift only annotated points (-1 sentinels stay -1)
    t[:, 4:14:2] = np.where(t[:, 4:14:2] >= 0, t[:, 4:14:2] + left, -1.0)
    t[:, 5:14:2] = np.where(t[:, 5:14:2] >= 0, t[:, 5:14:2] + top, -1.0)
    return canvas, t


class ExtendedPreproc:
    def __init__(self, img_dim, rgb_means, expand_prob=0.5, expand_max=2.0,
                 extras_prob=0.01):
        self.inner = preproc(img_dim, rgb_means)
        self.rgb_means = rgb_means
        self.expand_prob = expand_prob
        self.expand_max = expand_max
        self.extras_prob = extras_prob

    def __call__(self, img, target):
        img = _photometric_extras(img, self.extras_prob)
        img = np.float32(img)
        target = np.asarray(target, dtype=np.float64).reshape(-1, 15)
        if self.expand_prob > 0 and random.random() < self.expand_prob:
            img, target = _expand(img, target, self.rgb_means, self.expand_max)
        return self.inner(img, target)
