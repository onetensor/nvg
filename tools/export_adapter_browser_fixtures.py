from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_tensor(path: Path, value: np.ndarray) -> None:
    np.ascontiguousarray(value.astype("<f4")).tofile(path)


def write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create adapter and end-to-end browser fixtures")
    parser.add_argument("--reference", required=True, help="NPZ emitted by tools.verify_adapter")
    parser.add_argument("--output", default="public/validation")
    parser.add_argument("--adapter-url", default="../../models/pseudo-raw-adapter.onnx")
    parser.add_argument("--sid-url", default="../../models/sid-sony.onnx")
    args = parser.parse_args()
    data = np.load(args.reference)
    input_value = data["input"].astype(np.float32)
    adapter_output = data["adapter_output"].astype(np.float32)
    end_to_end_output = data["end_to_end_output"].astype(np.float32)
    gain = float(data["gain"])
    root = Path(args.output)

    adapter = root / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    write_tensor(adapter / "input.f32", input_value)
    write_tensor(adapter / "output.f32", adapter_output)
    write_manifest(adapter / "manifest.json", {
        "name": "Adapter",
        "inputUrl": "./validation/adapter/input.f32",
        "outputUrl": "./validation/adapter/output.f32",
        "inputShape": list(input_value.shape),
        "outputShape": list(adapter_output.shape),
        "stages": [{
            "modelUrl": args.adapter_url,
            "inputName": "input",
            "outputName": "pseudo_raw",
        }],
    })

    end_to_end = root / "end-to-end"
    end_to_end.mkdir(parents=True, exist_ok=True)
    write_tensor(end_to_end / "input.f32", input_value)
    write_tensor(end_to_end / "output.f32", end_to_end_output)
    write_manifest(end_to_end / "manifest.json", {
        "name": "Adapter + SID",
        "inputUrl": "./validation/end-to-end/input.f32",
        "outputUrl": "./validation/end-to-end/output.f32",
        "inputShape": list(input_value.shape),
        "outputShape": list(end_to_end_output.shape),
        "stages": [{
            "modelUrl": args.adapter_url,
            "inputName": "input",
            "outputName": "pseudo_raw",
            "gainClamp": gain,
        }, {
            "modelUrl": args.sid_url,
            "inputName": "input",
            "outputName": "output",
        }],
    })
    print(json.dumps({"adapter": str(adapter), "end_to_end": str(end_to_end), "gain": gain}, indent=2))


if __name__ == "__main__":
    main()
