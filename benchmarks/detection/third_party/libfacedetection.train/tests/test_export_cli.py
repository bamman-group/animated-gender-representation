from __future__ import annotations

import argparse
from pathlib import Path
from shutil import rmtree

import onnx
import torch

from yunet_train.cli.export_cpp import export_cpp
from yunet_train.cli.export_onnx import export_onnx
from yunet_train.tasks.face import build_yunet


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "export_cli"


def _checkpoint(path: Path) -> Path:
    model = build_yunet("yunet_s")
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "config": {"variant": "yunet_s"},
            "metrics": {},
        },
        path,
    )
    return path


def test_export_onnx_smoke() -> None:
    work_dir = OUTPUT_ROOT / "onnx"
    if work_dir.exists():
        rmtree(work_dir)
    work_dir.mkdir(parents=True)
    checkpoint = _checkpoint(work_dir / "yunet_s.pth")
    output_file = work_dir / "yunet_s.onnx"

    result = export_onnx(
        argparse.Namespace(
            checkpoint=checkpoint,
            variant="yunet_s",
            output_file=output_file,
            shape=[64, 64],
            opset_version=11,
            dynamic_export=False,
            verify=True,
            device="cpu",
        )
    )

    assert result == output_file
    model = onnx.load(str(output_file))
    assert len(model.graph.output) == 12
    assert [output.name for output in model.graph.output][:3] == ["cls_8", "cls_16", "cls_32"]
    rmtree(work_dir)


def test_export_cpp_smoke() -> None:
    work_dir = OUTPUT_ROOT / "cpp"
    if work_dir.exists():
        rmtree(work_dir)
    work_dir.mkdir(parents=True)
    checkpoint = _checkpoint(work_dir / "yunet_s.pth")
    output_file = work_dir / "facedetectcnn-data.cpp"

    result = export_cpp(
        argparse.Namespace(
            checkpoint=checkpoint,
            variant="yunet_s",
            output_file=output_file,
            precision=".3g",
        )
    )

    assert result == output_file
    text = output_file.read_text(encoding="utf-8")
    assert '#include "facedetectcnn.h"' in text
    assert "ConvInfoStruct param_pConvInfo" in text
    assert "backbone__model0_pw_weight" in text
    rmtree(work_dir)
