from __future__ import annotations

import argparse
from pathlib import Path

import torch

from yunet_train.engine.onnx_export import export_model_to_onnx, parse_input_shape
from yunet_train.tasks.face import build_yunet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight YuNet checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--shape", type=int, nargs="+", default=[640, 640])
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--dynamic-export", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = export_onnx(args)
    print(f"Successfully exported ONNX model: {output_file}")


def export_onnx(args: argparse.Namespace) -> Path:
    input_shape = parse_input_shape(args.shape)
    return export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        build_model=build_yunet,
        output_file=args.output_file or _default_output_file(args.checkpoint, args.variant, input_shape, args.dynamic_export),
        input_shape=input_shape,
        output_names=_output_names(),
        flatten_outputs=_flatten_export_outputs,
        variant=args.variant,
        device=args.device,
        opset_version=args.opset_version,
        dynamic_export=args.dynamic_export,
        verify=args.verify,
    )


def _default_output_file(
    checkpoint: Path,
    variant: str | None,
    input_shape: tuple[int, int, int, int],
    dynamic_export: bool,
) -> Path:
    tag = "dynamic" if dynamic_export else f"{input_shape[-2]}_{input_shape[-1]}"
    variant_tag = variant or "auto"
    return Path("work_dirs") / "export" / f"{checkpoint.stem}_{variant_tag}_{tag}.onnx"


def _output_names() -> list[str]:
    names: list[str] = []
    for head in ("cls", "obj", "bbox", "kps"):
        names.extend([f"{head}_{stride}" for stride in (8, 16, 32)])
    return names


def _flatten_export_outputs(model: torch.nn.Module, image: torch.Tensor) -> list[torch.Tensor]:
    cls_scores, bbox_preds, objectnesses, kps_preds = model(image)
    batch_size = image.shape[0]
    cls = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid() for pred in cls_scores]
    obj = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid() for pred in objectnesses]
    bbox = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for pred in bbox_preds]
    kps = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 10) for pred in kps_preds]
    return cls + obj + bbox + kps


if __name__ == "__main__":
    main()
