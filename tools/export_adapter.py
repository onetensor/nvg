from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.adapter_model import PseudoRawAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/pseudo-raw-adapter.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PseudoRawAdapter(
        width=int(payload.get("width", 32)),
        residual_scale=float(payload.get("residual_scale", 0.25)),
    ).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand(1, 3, 128, 128),
        output,
        input_names=["input"],
        output_names=["pseudo_raw"],
        dynamic_axes={
            "input": {0: "batch", 2: "rgb_height", 3: "rgb_width"},
            "pseudo_raw": {0: "batch", 2: "raw_height", 3: "raw_width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(json.dumps({"output": str(output), "parameter_count": model.parameter_count}, indent=2))


if __name__ == "__main__":
    main()
