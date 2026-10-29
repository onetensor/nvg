from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="TensorFlow reference NPZ with NHWC input/output")
    parser.add_argument("--output", default="public/validation/sid")
    parser.add_argument("--model-url", default="../../models/sid-sony.onnx")
    args = parser.parse_args()
    data = np.load(args.reference)
    inputs = np.ascontiguousarray(data["input"].transpose(0, 3, 1, 2).astype("<f4"))
    outputs = np.ascontiguousarray(data["output"].transpose(0, 3, 1, 2).astype("<f4"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inputs.tofile(output / "input.f32")
    outputs.tofile(output / "output.f32")
    manifest = {
        "name": "SID",
        "inputUrl": "./validation/sid/input.f32",
        "outputUrl": "./validation/sid/output.f32",
        "inputShape": list(inputs.shape),
        "outputShape": list(outputs.shape),
        "stages": [{
            "modelUrl": args.model_url,
            "inputName": "input",
            "outputName": "output",
        }],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
