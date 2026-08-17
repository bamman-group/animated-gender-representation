from __future__ import annotations

import argparse
import os
from pathlib import Path

import yunet_train.cli.export_tflite as export_tflite_cli


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "export_tflite_cli"


def test_export_tflite_wraps_onnx2tf(monkeypatch) -> None:
    work_root = OUTPUT_ROOT / f"case_{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    def fake_export_onnx(args: argparse.Namespace) -> Path:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_bytes(b"fake onnx")
        return args.output_file

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert check is True
        output_dir = Path(command[command.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model_float32.tflite").write_bytes(b"fake tflite")

    monkeypatch.setattr(export_tflite_cli, "_ensure_onnx2tf_available", lambda command: None)
    monkeypatch.setattr(export_tflite_cli, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(export_tflite_cli.subprocess, "run", fake_run)

    output_file = work_root / "yunet.tflite"
    result = export_tflite_cli.export_tflite(
        argparse.Namespace(
            checkpoint=Path("unused.pth"),
            variant="yunet_s",
            output_file=output_file,
            shape=[64, 64],
            onnx_file=None,
            work_dir=work_root / "build",
            keep_intermediate=True,
            onnx2tf_command="onnx2tf",
            extra_onnx2tf_args=["-ois", "input:1,3,64,64"],
        )
    )

    assert result == output_file
    assert output_file.read_bytes() == b"fake tflite"
    assert calls[0][0] == "onnx2tf"
    assert calls[0][calls[0].index("-i") + 1] == str(work_root / "build" / "yunet.onnx")
    assert "-tb" in calls[0]
    assert "flatbuffer_direct" in calls[0]
    assert (work_root / "build" / "yunet_onnx2tf").exists()
