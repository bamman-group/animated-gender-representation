from pathlib import Path

import torchvision.transforms as T
from PIL import Image

IMAGE_SIZE = 256

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
    bboxes = {}
    with open(det_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            rel_path, x1, y1, x2, y2 = parts
            bboxes[rel_path] = (int(x1), int(y1), int(x2), int(y2))
    return bboxes
