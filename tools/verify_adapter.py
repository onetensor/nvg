from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.adapter_model import PseudoRawAdapter
from training.sid_model import load_sid


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = np.square(delta).mean()
    return {
        "mae": float(np.abs(delta).mean()),
        "max_error": float(np.abs(delta).max()),
        "psnr": float(-10 * math.log10(max(mse, 1e-16))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch ↔ ONNX adapter and end-to-end equivalence")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--adapter-onnx", required=True)
    parser.add_argument("--sid-pytorch", required=True)
    parser.add_argument("--sid-onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture", help="optional NPZ of PyTorch references for browser validation")
    parser.add_argument("--gain", type=float, default=30.0)
    parser.add_argument("--max-adapter-mae", type=float, default=1e-5)
    parser.add_argument("--max-end-to-end-mae", type=float, default=5e-5)
    args = parser.parse_args()
    rng = np.random.default_rng(1337)
    # Exercise the low-light, non-saturated regime used by SID.
    input_value = rng.random((1, 3, 128, 128), dtype=np.float32) * 0.025
    payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    adapter = PseudoRawAdapter(
        width=int(payload.get("width", 32)),
        residual_scale=float(payload.get("residual_scale", 0.25)),
    ).eval()
    adapter.load_state_dict(payload["state_dict"], strict=True)
    sid = load_sid(args.sid_pytorch).eval()
    with torch.inference_mode():
        torch_raw = adapter(torch.from_numpy(input_value)).numpy()
        torch_rgb = sid(torch.from_numpy(np.clip(torch_raw * args.gain, 0, 1))).numpy()
    adapter_session = ort.InferenceSession(args.adapter_onnx, providers=["CPUExecutionProvider"])
    sid_session = ort.InferenceSession(args.sid_onnx, providers=["CPUExecutionProvider"])
    onnx_raw = adapter_session.run(["pseudo_raw"], {"input": input_value})[0]
    onnx_rgb = sid_session.run(["output"], {"input": np.clip(onnx_raw * args.gain, 0, 1)})[0]
    report = {
        "adapter": metrics(onnx_raw, torch_raw),
        "end_to_end": metrics(onnx_rgb, torch_rgb),
        "gain": args.gain,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.fixture:
        fixture = Path(args.fixture)
        fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fixture,
            input=input_value,
            adapter_output=torch_raw,
            end_to_end_output=torch_rgb,
            gain=np.float32(args.gain),
        )
    print(json.dumps(report, indent=2))
    if report["adapter"]["mae"] > args.max_adapter_mae:
        raise SystemExit("Adapter ONNX export exceeded its equivalence gate")
    if report["end_to_end"]["mae"] > args.max_end_to_end_mae:
        raise SystemExit("Adapter + SID ONNX pipeline exceeded its equivalence gate")


if __name__ == "__main__":
    main()
