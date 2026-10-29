from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--sample", help="optional NPZ containing key 'input'")
    parser.add_argument("--sample-size", type=int, default=128, help="center crop for NPZ samples; 0 keeps full size")
    parser.add_argument("--scale", type=float, default=1.0, help="scale deterministic random input")
    parser.add_argument("--max-mae", type=float, default=0.002)
    parser.add_argument("--max-error", type=float, default=0.03)
    args = parser.parse_args()
    if args.sample:
        sample = np.load(args.sample)["input"].astype(np.float32)
        if sample.ndim != 4:
            raise ValueError("sample input must be rank-4")
        if sample.shape[1] != args.channels and sample.shape[-1] == args.channels:
            sample = sample.transpose(0, 3, 1, 2)
        if sample.shape[1] != args.channels:
            raise ValueError(f"sample has shape {sample.shape}, expected {args.channels} channels")
        if args.sample_size:
            size = min(args.sample_size, sample.shape[2], sample.shape[3])
            y = (sample.shape[2] - size) // 2
            x = (sample.shape[3] - size) // 2
            sample = sample[:, :, y:y + size, x:x + size]
    else:
        height = args.height or args.size
        width = args.width or args.size
        sample = np.random.default_rng(1337).random(
            (1, args.channels, height, width), dtype=np.float32,
        ) * args.scale
    baseline = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"]).run(
        [args.output_name], {args.input_name: sample},
    )[0]
    candidate = ort.InferenceSession(args.fp16, providers=["CPUExecutionProvider"]).run(
        [args.output_name], {args.input_name: sample},
    )[0]
    difference = np.abs(candidate.astype(np.float64) - baseline.astype(np.float64))
    report = {"mae": float(difference.mean()), "max_error": float(difference.max())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["mae"] > args.max_mae or report["max_error"] > args.max_error:
        raise SystemExit("FP16 conversion exceeded its numerical error gate")


if __name__ == "__main__":
    main()
