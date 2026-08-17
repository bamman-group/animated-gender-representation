from __future__ import annotations

import argparse
from pathlib import Path
from shutil import rmtree

import onnx
import torch

from yunet_train.cli.export_pose_onnx import export_pose_onnx
from yunet_train.tasks.pose import build_yunet_pose


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "export_pose_cli"


def _checkpoint(path: Path) -> Path:
    model = build_yunet_pose("yunet_n", kpt_shape=(17, 3))
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "config": {"variant": "yunet_n"},
            "metrics": {},
        },
        path,
    )
    return path


def test_export_pose_onnx_smoke() -> None:
    work_dir = OUTPUT_ROOT / "onnx"
    if work_dir.exists():
        rmtree(work_dir)
    work_dir.mkdir(parents=True)
    checkpoint = _checkpoint(work_dir / "yunet_pose_n.pth")
    output_file = work_dir / "yunet_pose_n.onnx"

    result = export_pose_onnx(
        argparse.Namespace(
            checkpoint=checkpoint,
            variant="yunet_n",
            output_file=output_file,
            shape=[64, 64],
            kpt_shape=[17, 3],
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
    assert [output.name for output in model.graph.output][3:6] == ["obj_8", "obj_16", "obj_32"]
    assert [output.name for output in model.graph.output][-3:] == ["kpt_8", "kpt_16", "kpt_32"]
    rmtree(work_dir)
