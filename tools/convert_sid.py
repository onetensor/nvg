from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.sid_model import SIDUNet


CONVOLUTIONS = [
    "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
    "conv4_1", "conv4_2", "conv5_1", "conv5_2", "conv6_1", "conv6_2",
    "conv7_1", "conv7_2", "conv8_1", "conv8_2", "conv9_1", "conv9_2", "conv10",
]
TF_NAMES = {name: f"g_{name}" for name in CONVOLUTIONS}


def convert(checkpoint: str, output: Path) -> SIDUNet:
    import tensorflow as tf

    reader = tf.train.load_checkpoint(checkpoint)
    available = {name: shape for name, shape in tf.train.list_variables(checkpoint)}
    model = SIDUNet()
    state = model.state_dict()
    for name, tf_name in TF_NAMES.items():
        weight_name = f"{tf_name}/weights"
        bias_name = f"{tf_name}/biases"
        if weight_name not in available or bias_name not in available:
            raise KeyError(f"Missing official variables {weight_name} / {bias_name}")
        state[f"{name}.weight"] = torch.from_numpy(reader.get_tensor(weight_name).transpose(3, 2, 0, 1).copy())
        state[f"{name}.bias"] = torch.from_numpy(reader.get_tensor(bias_name).copy())

    roots = []
    for variable, shape in available.items():
        if "/" not in variable and tuple(shape) in {(2, 2, 256, 512), (2, 2, 128, 256),
                                                     (2, 2, 64, 128), (2, 2, 32, 64)}:
            roots.append(variable)
    roots.sort(key=lambda value: 0 if value == "Variable" else int(value.rsplit("_", 1)[-1]) + 1)
    if len(roots) != 4:
        raise ValueError(f"Expected four anonymous transpose-convolution variables, found {roots}")
    for layer, variable in zip(("up6", "up7", "up8", "up9"), roots, strict=True):
        # TF [height, width, output_channels, input_channels] → torch [input, output, h, w]
        state[f"{layer}.weight"] = torch.from_numpy(reader.get_tensor(variable).transpose(3, 2, 0, 1).copy())

    # TensorFlow NHWC depth_to_space uses channels in [offset_y, offset_x,
    # output_channel] order. PyTorch pixel_shuffle expects
    # [output_channel, offset_y, offset_x]. Reorder the final 12 filters once
    # so the exported graph can use the standard PixelShuffle/DepthToSpace op.
    depth_to_space_order = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    state["conv10.weight"] = state["conv10.weight"][depth_to_space_order]
    state["conv10.bias"] = state["conv10.bias"][depth_to_space_order]
    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "source_checkpoint": checkpoint,
        "architecture": "official Sony SID U-Net",
        "tensorflow_variables": sorted(TF_NAMES.values()) + roots,
    }, output)
    return model.eval()


def export_onnx(model: SIDUNet, output: Path, opset: int) -> None:
    sample = torch.rand(1, 4, 64, 64)
    onnx_path = output.with_suffix(".onnx")
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "raw_height", 3: "raw_width"},
            "output": {0: "batch", 2: "rgb_height", 3: "rgb_width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(json.dumps({"pytorch": str(output), "onnx": str(onnx_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official TensorFlow Sony SID checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint prefix, e.g. checkpoint/Sony/model.ckpt")
    parser.add_argument("--output", default="models/sid-sony.pt")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    output = Path(args.output)
    model = convert(args.checkpoint, output)
    export_onnx(model, output, args.opset)


if __name__ == "__main__":
    main()
