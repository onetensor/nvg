from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.adapter_model import PseudoRawAdapter
from training.sid_model import load_sid


def export(
    model: torch.nn.Module,
    sample: torch.Tensor,
    output: Path,
    output_name: str,
    opset: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["input"],
        output_names=[output_name],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"STATIC_ONNX_EXPORTED {output} bytes={output.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-shape SID and adapter graphs for WebGPU capture")
    parser.add_argument("--sid-checkpoint", required=True)
    parser.add_argument("--sid-output", required=True)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--adapter-output")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("Full-resolution static dimensions must be multiples of 32")

    sid = load_sid(args.sid_checkpoint)
    export(
        sid,
        torch.zeros(1, 4, args.height // 2, args.width // 2),
        Path(args.sid_output),
        "output",
        args.opset,
    )

    if bool(args.adapter_checkpoint) != bool(args.adapter_output):
        raise ValueError("--adapter-checkpoint and --adapter-output must be provided together")
    if args.adapter_checkpoint and args.adapter_output:
        payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        adapter = PseudoRawAdapter(
            width=int(payload.get("width", 64)),
            residual_scale=float(payload.get("residual_scale", 0.10)),
        )
        adapter.load_state_dict(payload["state_dict"], strict=True)
        export(
            adapter,
            torch.zeros(1, 3, args.height, args.width),
            Path(args.adapter_output),
            "pseudo_raw",
            args.opset,
        )


if __name__ == "__main__":
    main()
