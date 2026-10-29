from __future__ import annotations

import argparse
from pathlib import Path

from tools.convert_sid import export_onnx
from training.sid_model import load_sid


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PyTorch SID checkpoint to dynamic-shape ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_sid(args.checkpoint).eval()
    # export_onnx derives the ONNX path by replacing this path's suffix.
    export_onnx(model, output.with_suffix(".pt"), args.opset)


if __name__ == "__main__":
    main()
