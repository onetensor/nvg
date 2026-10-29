from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.sid_model import load_sid


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    mse = np.square(candidate.astype(np.float64) - reference.astype(np.float64)).mean()
    return {
        "mae": float(difference.mean()),
        "max_error": float(difference.max()),
        "psnr": float(-10 * math.log10(max(mse, 1e-16))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.reference)
    input_nhwc = data["input"].astype(np.float32)
    reference = data["output"].astype(np.float32)
    input_nchw = input_nhwc.transpose(0, 3, 1, 2)
    model = load_sid(args.pytorch).eval()
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(input_nchw)).numpy().transpose(0, 2, 3, 1)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_output = session.run(["output"], {"input": input_nchw})[0].transpose(0, 2, 3, 1)
    report = {
        "tensorflow_vs_pytorch": metrics(torch_output, reference),
        "tensorflow_vs_onnx": metrics(onnx_output, reference),
        "pytorch_vs_onnx": metrics(onnx_output, torch_output),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    comparison = np.concatenate([reference[0], torch_output[0], onnx_output[0]], axis=1)
    Image.fromarray((np.clip(comparison, 0, 1) * 255).astype(np.uint8)).save(output / "comparison.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
