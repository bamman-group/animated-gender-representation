from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from yunet_train.tasks.face import build_yunet
from yunet_train.engine import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YuNet checkpoint to libfacedetection C++ data.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--output-file", type=Path, default=Path("work_dirs/export/facedetectcnn-data.cpp"))
    parser.add_argument("--precision", default=".3g")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = export_cpp(args)
    print(f"Convert successful: {output_file}")


def export_cpp(args: argparse.Namespace) -> Path:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_yunet(variant)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.eval()

    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(CppConvertor(model, precision=args.precision).data, encoding="utf-8")
    return output_file


def data2str_as_precision(data: float, precision: str) -> str:
    text = format(float(data), precision)
    if "." not in text and "e" not in text:
        return text + ".f"
    return text + "f"


class CppConvertor:
    """Export YuNet convolution weights in libfacedetection's C++ data layout."""

    def __init__(self, model: nn.Module, *, precision: str = ".3g"):
        model.eval()
        self.support_modules = ("Conv_head", "ConvDPUnit", "Conv4layerBlock")
        self.precision = precision
        self.module_list: list[dict[str, object]] = []
        self.cppdata: list[dict[str, object]] = []
        self.data = (
            "// Auto generated data file\n"
            "// Copyright (c) 2018-2023, Shiqi Yu, all rights reserved.\n"
            '#include "facedetectcnn.h"\n\n'
        )
        self.loop_search_modules(model)
        self.convert()

    @staticmethod
    def combine_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        conv_result = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        scales = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv_result.bias.data = (conv_bias.data - bn.running_mean) * scales + bn.bias.data
        for channel in range(conv.out_channels):
            conv_result.weight.data[channel, ...] = conv.weight.data[channel, ...] * scales[channel]
        return conv_result

    def convert_param2string(
        self,
        conv: nn.Conv2d,
        name: str,
        *,
        is_depthwise: bool = False,
        with_bn_relu: bool = False,
        is_first_3x3x3: bool = False,
    ) -> dict[str, object]:
        out_channels, in_channels, width, height = conv.weight.size()

        if is_first_3x3x3:
            weight = conv.weight.detach().cpu().numpy().reshape((-1, 27))
            reordered = weight.copy()
            for idx in range(out_channels):
                for offset in range(27):
                    weight[idx, (offset % 9) * 3 + offset // 9] = reordered[idx, offset]
            weight = np.hstack((weight, np.zeros((out_channels, 5)))).reshape(-1)
        elif is_depthwise:
            weight = conv.weight.detach().cpu().numpy().reshape((-1, 9)).transpose().reshape(-1)
        else:
            weight = conv.weight.detach().cpu().numpy().reshape(-1)

        bias_tensor = conv.bias if conv.bias is not None else torch.zeros(out_channels)
        bias = bias_tensor.detach().cpu().numpy().reshape(-1)
        weight_size = (
            f"{out_channels}*32*1*1"
            if is_first_3x3x3
            else f"{out_channels}*{in_channels}*{width}*{height}"
        )
        in_channels_for_struct = 32 if is_first_3x3x3 else out_channels if is_depthwise else in_channels
        return {
            "type": "float",
            "weight_name": f"{name}_weight",
            "weight_size": weight_size,
            "weight": ",".join(data2str_as_precision(value, self.precision) for value in weight),
            "bias_name": f"{name}_bias",
            "bias_size": str(out_channels),
            "bias": ",".join(data2str_as_precision(value, self.precision) for value in bias),
            "with_bn": with_bn_relu,
            "is_dw": is_depthwise,
            "in_channels": in_channels_for_struct,
            "out_channels": out_channels,
        }

    def convert_module2string(self, conv: nn.Module, name: str, module_type: str) -> None:
        if module_type == "Conv_head":
            self.cppdata.append(
                self.convert_param2string(
                    self.combine_conv_bn(conv.conv1, conv.bn1),
                    name + "_pw",
                    with_bn_relu=True,
                    is_first_3x3x3=True,
                )
            )
            self.convert_module2string(conv.conv2, name + "_dp", "ConvDPUnit")
        elif module_type == "ConvDPUnit":
            self.cppdata.append(self.convert_param2string(conv.conv1, name + "_pw"))
            if conv.withBNRelu:
                self.cppdata.append(
                    self.convert_param2string(
                        self.combine_conv_bn(conv.conv2, conv.bn),
                        name + "_dw",
                        is_depthwise=True,
                        with_bn_relu=True,
                    )
                )
            else:
                self.cppdata.append(self.convert_param2string(conv.conv2, name + "_dw", is_depthwise=True))
        elif module_type == "Conv4layerBlock":
            self.convert_module2string(conv.conv1, name + "_dp1", "ConvDPUnit")
            self.convert_module2string(conv.conv2, name + "_dp2", "ConvDPUnit")
        else:
            raise ValueError(f"Unsupported module: {name} ({module_type})")

    def loop_search_modules(self, model: nn.Module, last_name: str = "") -> None:
        for name, module in model.named_children():
            module_name = f"{last_name}__{name}"
            module_type = module.__class__.__name__
            if module_type in self.support_modules:
                self.module_list.append({"type": module_type, "name": module_name[2:], "module": module})
            else:
                self.loop_search_modules(module, module_name)

    @staticmethod
    def python_bool_to_c_bool(value: bool) -> str:
        return "true" if value else "false"

    def convert(self) -> None:
        for module in self.module_list:
            self.convert_module2string(
                conv=module["module"],
                name=str(module["name"]),
                module_type=str(module["type"]),
            )

        for item in self.cppdata:
            self.data += f"{item['type']} {item['weight_name']}[{item['weight_size']}] = {{{item['weight']}}};\n"
            self.data += f"{item['type']} {item['bias_name']}[{item['bias_size']}] = {{{item['bias']}}};\n"

        self.data += "\n//(in_channels, out_channels, is_depthwise, is_pointwise, with_bn, weight_ptr, bias_ptr)\n"
        self.data += f"ConvInfoStruct param_pConvInfo[{len(self.cppdata)}] = {{\n"
        for idx, item in enumerate(self.cppdata):
            suffix = "," if idx < len(self.cppdata) - 1 else ""
            self.data += (
                f"\t{{{item['in_channels']}, {item['out_channels']}, "
                f"{self.python_bool_to_c_bool(bool(item['is_dw']))}, "
                f"{self.python_bool_to_c_bool(not bool(item['is_dw']))}, "
                f"{self.python_bool_to_c_bool(bool(item['with_bn']))}, "
                f"{item['weight_name']}, {item['bias_name']}}}{suffix}\n"
            )
        self.data += "};"


if __name__ == "__main__":
    main()
