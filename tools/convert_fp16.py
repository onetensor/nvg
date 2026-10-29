from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX weights/activations to FP16 with FP32 model I/O")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    converted = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, output)
    print(f"FP16_CONVERTED {output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
