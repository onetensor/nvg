from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from training.outline_net import OutlinedSID, load_outline
from training.sid_model import load_sid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a mono direct SID fused with its outline network: [luma, outline logits]")
    parser.add_argument("--sid", required=True)
    parser.add_argument("--outline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    model = OutlinedSID(load_sid(args.sid), load_outline(args.outline)).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, 13, 64, 64),
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
    # The runtime's portrait 360p shape.
    sample = torch.rand(1, 13, 320, 192) * 0.6
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        ["output"], {"input": sample.numpy()})[0]
    report = {
        "onnx": str(output),
        "bytes": output.stat().st_size,
        "output_shape": list(actual.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "max_abs_error": float(np.abs(actual - expected).max()),
    }
    print("OUTLINED_EXPORT " + json.dumps(report), flush=True)
    if report["max_abs_error"] > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
