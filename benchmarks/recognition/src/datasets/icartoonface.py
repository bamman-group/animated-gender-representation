from pathlib import Path

import torchvision.transforms as T
from PIL import Image

# Images resized to 256x256 before/after cropping. Used for evaluation
# (src/evaluate.py, src/evaluate_film.py) - fine-tuning scripts
# (src/train_dino.py, src/train_buffalo.py) use their own model-specific
# transforms/resolutions instead.
IMAGE_SIZE = 256

# Fraction of the bbox's own width/height added on *each* side before
# cropping (extends into the real surrounding image, not blank padding),
# clipped to the image bounds. 0.0 = tight crop, no padding. 0.25 = 25% of
# the bbox width/height added on every side (so a 100x100 bbox becomes a
# 150x150 crop, still centered on the original box).
DEFAULT_CROP_PADDING = 0.0

EVAL_TRANSFORM = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def pad_bbox(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int, padding: float
) -> tuple[int, int, int, int]:
    """Grows a bbox by `padding` fraction of its own width/height on each
    side (e.g. padding=0.25 adds 25% of the width on the left and another
    25% on the right, so total width becomes 1.5x), clipped to the image
    bounds."""
    x1, y1, x2, y2 = bbox
    pad_w, pad_h = (x2 - x1) * padding, (y2 - y1) * padding

    new_x1 = max(0, int(round(x1 - pad_w)))
    new_y1 = max(0, int(round(y1 - pad_h)))
    new_x2 = min(image_width, int(round(x2 + pad_w)))
    new_y2 = min(image_height, int(round(y2 + pad_h)))
    return new_x1, new_y1, new_x2, new_y2


def crop_face(
    image: Image.Image, bbox: tuple[int, int, int, int], padding: float = DEFAULT_CROP_PADDING
) -> Image.Image:
    padded = pad_bbox(bbox, image.width, image.height, padding)
    return image.crop(padded)


def parse_det_file(det_file: Path) -> dict[str, tuple[int, int, int, int]]:
    """Parses a "<relative_path>\\tx1\\ty1\\tx2\\ty2" bbox file, e.g.
    icartoonface_rectrain_det.txt, keyed by relative image path."""
    bboxes = {}
    with open(det_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            rel_path, x1, y1, x2, y2 = parts
            bboxes[rel_path] = (int(x1), int(y1), int(x2), int(y2))
    return bboxes
