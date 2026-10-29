from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def run(path: str, output: str, value: np.ndarray) -> np.ndarray:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run([output], {"input": value})[0]


def error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {"mae": float(difference.mean()), "max_error": float(difference.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-shape graphs with validated dynamic ONNX graphs")
    parser.add_argument("--adapter-dynamic", required=True)
    parser.add_argument("--adapter-static", required=True)
    parser.add_argument("--sid-dynamic", required=True)
    parser.add_argument("--sid-static", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--gain", type=float, default=30.0)
    parser.add_argument("--max-mae", type=float, default=1e-6)
    args = parser.parse_args()

    inputs = np.random.default_rng(1337).random(
        (1, 3, args.height, args.width), dtype=np.float32,
    ) * 0.025
    dynamic_raw = run(args.adapter_dynamic, "pseudo_raw", inputs)
    static_raw = run(args.adapter_static, "pseudo_raw", inputs)
    dynamic_rgb = run(args.sid_dynamic, "output", np.clip(dynamic_raw * args.gain, 0, 1))
    static_rgb = run(args.sid_static, "output", np.clip(static_raw * args.gain, 0, 1))
    report = {
        "shape": [1, 3, args.height, args.width],
        "adapter": error(static_raw, dynamic_raw),
        "end_to_end": error(static_rgb, dynamic_rgb),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["adapter"]["mae"] > args.max_mae or report["end_to_end"]["mae"] > args.max_mae:
        raise SystemExit("Fixed-shape export exceeded its dynamic-graph equivalence gate")


if __name__ == "__main__":
    main()
