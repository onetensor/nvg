from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import pack_sony


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed official-RAW SID validation tensor")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--ratio", type=float, default=100.0)
    parser.add_argument("--size", type=int, default=256, help="Packed-domain square crop, divisible by 16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.size % 16:
        raise ValueError("--size must be divisible by 16")
    with rawpy.imread(args.raw) as raw:
        packed = pack_sony(raw)
    _, height, width = packed.shape
    y = max(0, (height - args.size) // 2) // 16 * 16
    x = max(0, (width - args.size) // 2) // 16 * 16
    sample = np.minimum(packed[:, y:y + args.size, x:x + args.size] * args.ratio, 1.0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input=sample.transpose(1, 2, 0)[None].astype(np.float32))


if __name__ == "__main__":
    main()
