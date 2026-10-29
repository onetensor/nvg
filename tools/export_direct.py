from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from training.sid_model import load_sid


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a direct (12-channel packed RGB) SID checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    model = load_sid(args.checkpoint).eval()
    if model.in_channels not in (12, 13):
        raise ValueError("Expected a direct (12- or 13-channel) checkpoint")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, model.in_channels, 64, 64),
        output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "raw_height", 3: "raw_width"},
            "output": {0: "batch", 2: "rgb_height", 3: "rgb_width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # The runtime's portrait 180p shape, so equivalence covers a non-square input.
    sample = torch.rand(1, model.in_channels, 160, 96) * 0.6
    with torch.inference_mode():
        expected = model(sample).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["output"], {"input": sample.numpy()})[0]
    report = {
        "onnx": str(output),
        "bytes": output.stat().st_size,
        "width": model.width,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "output_shape": list(actual.shape),
        "max_abs_error": float(np.abs(actual - expected).max()),
    }
    print("DIRECT_EXPORT " + json.dumps(report), flush=True)
    if report["max_abs_error"] > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
