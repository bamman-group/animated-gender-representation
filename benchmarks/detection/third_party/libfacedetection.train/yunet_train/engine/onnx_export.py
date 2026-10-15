from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint


def export_model_to_onnx(
    *,
    checkpoint_path: Path,
    build_model: Callable[[str], torch.nn.Module],
    output_file: Path,
    input_shape: tuple[int, int, int, int],
    output_names: list[str],
    flatten_outputs: Callable[[torch.nn.Module, torch.Tensor], list[torch.Tensor]],
    variant: str | None,
    device: torch.device | str,
    opset_version: int,
    dynamic_export: bool,
    verify: bool,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_variant = variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_model(resolved_variant)
    load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    model.to(device).eval()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    example_input = torch.randn(input_shape, dtype=torch.float32, device=device)
    dynamic_axes = _dynamic_axes(output_names) if dynamic_export else None

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_file),
            input_names=["input"],
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    check_onnx(output_file)
    if verify:
        verify_onnx(model, example_input, output_file, flatten_outputs)
    return output_file


def parse_input_shape(shape: list[int]) -> tuple[int, int, int, int]:
    if len(shape) == 1:
        height = width = shape[0]
    elif len(shape) == 2:
        height, width = shape
    else:
        raise ValueError("--shape expects one int or two ints")
    return (1, 3, height, width)


def check_onnx(output_file: Path) -> None:
    import onnx

    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)

    inputs = model.graph.input
    name_to_input = {graph_input.name: graph_input for graph_input in inputs}
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(model, str(output_file))


def verify_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_file: Path,
    flatten_outputs: Callable[[torch.nn.Module, torch.Tensor], list[torch.Tensor]],
) -> None:
    import onnxruntime

    with torch.no_grad():
        torch_outputs = [output.detach().cpu().numpy() for output in flatten_outputs(model, example_input)]
    session = onnxruntime.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: example_input.detach().cpu().numpy()})
    if len(torch_outputs) != len(onnx_outputs):
        raise AssertionError(f"ONNX output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}")
    for idx, (torch_output, onnx_output) in enumerate(zip(torch_outputs, onnx_outputs)):
        np.testing.assert_allclose(
            onnx_output,
            torch_output,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"ONNX output {idx} differs from PyTorch",
        )
    print("The numerical values are close between PyTorch and ONNX")


def _dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    dynamic_axes = {name: {0: "batch", 1: "dim"} for name in output_names}
    dynamic_axes["input"] = {0: "batch", 2: "height", 3: "width"}
    return dynamic_axes
