from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from yunet_train.cli.export_onnx import export_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YuNet to TFLite through ONNX + onnx2tf flatbuffer_direct."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--output-file", type=Path, default=Path("work_dirs/export/yunet.tflite"))
    parser.add_argument("--shape", type=int, nargs="+", default=[640, 640])
    parser.add_argument("--onnx-file", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/export/tflite_build"))
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--onnx2tf-command", default="onnx2tf")
    parser.add_argument("--extra-onnx2tf-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = export_tflite(args)
    print(f"Successfully exported TFLite model: {output_file}")


def export_tflite(args: argparse.Namespace) -> Path:
    _ensure_onnx2tf_available(args.onnx2tf_command)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx_file is None:
        onnx_file = args.work_dir / f"{args.output_file.stem}.onnx"
        export_onnx(
            argparse.Namespace(
                checkpoint=args.checkpoint,
                variant=args.variant,
                output_file=onnx_file,
                shape=args.shape,
                opset_version=11,
                dynamic_export=False,
                verify=False,
                device="cpu",
            )
        )
    else:
        onnx_file = args.onnx_file

    temp_output = args.work_dir / f"{args.output_file.stem}_onnx2tf_tmp"
    if temp_output.exists():
        shutil.rmtree(temp_output)
    command = [
        args.onnx2tf_command,
        "-i",
        str(onnx_file),
        "-o",
        str(temp_output),
        "-tb",
        "flatbuffer_direct",
        *args.extra_onnx2tf_args,
    ]
    subprocess.run(command, check=True)
    tflite_file = _find_tflite_file(temp_output)
    shutil.copyfile(tflite_file, args.output_file)
    if args.keep_intermediate:
        keep_dir = args.work_dir / f"{args.output_file.stem}_onnx2tf"
        if keep_dir.exists():
            shutil.rmtree(keep_dir)
        shutil.copytree(temp_output, keep_dir)
    else:
        shutil.rmtree(temp_output)

    return args.output_file


def _ensure_onnx2tf_available(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            "onnx2tf is not installed or not on PATH. "
            "Install it in an optional conversion environment, for example: "
            "`python -m pip install onnx2tf ai-edge-litert`."
        )


def _find_tflite_file(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*_float32.tflite"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*.tflite"))
    if not candidates:
        raise FileNotFoundError(f"onnx2tf did not produce a .tflite file under {output_dir}")
    return candidates[0]


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
