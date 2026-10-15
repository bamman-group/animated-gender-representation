from __future__ import annotations

import argparse
from pathlib import Path

import torch

from yunet_train.tasks.face import build_yunet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight YuNet forward check.")
    parser.add_argument("--variant", default="yunet_n", choices=("yunet_n", "yunet_s"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"missing keys: {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")
    if missing:
        print("first missing keys:", missing[:10])
    if unexpected:
        print("first unexpected keys:", unexpected[:10])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = build_yunet(args.variant).to(device).eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)

    torch.manual_seed(20260503)
    image = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        cls_scores, bbox_preds, objectnesses, kps_preds = model(image)

    for name, outputs in (
        ("cls", cls_scores),
        ("bbox", bbox_preds),
        ("obj", objectnesses),
        ("kps", kps_preds),
    ):
        shapes = [tuple(output.shape) for output in outputs]
        print(f"{name}: {shapes}")


if __name__ == "__main__":
    main()

