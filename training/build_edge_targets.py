"""Teacher outlines for every cached burst crop (see build_burst_dataset).

HED (soft output) runs on the clean long exposure after brightening it the way
the runtime's auto gain would (median luminance to 0.18), because the app shows
brightened output and HED misses edges in dim renderings. Each crop gets a
uint8 512x512 map at ``edges/<split>/<crop>.npy``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.phone_sim import libraw_decode, srgb_encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute HED outline targets for cached bursts")
    parser.add_argument("--data", required=True, help="Directory holding bursts/<split>/*.npz")
    args = parser.parse_args()
    from controlnet_aux import HEDdetector

    detector = HEDdetector.from_pretrained("lllyasviel/Annotators").to("cuda")
    root = Path(args.data)
    for split in ("train", "val", "test"):
        files = sorted(path for path in (root / "bursts" / split).glob("*.npz")
                       if not path.name.endswith(".partial.npz"))
        destination = root / "edges" / split
        destination.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(files):
            target = destination / path.with_suffix(".npy").name
            if target.exists():
                continue
            with np.load(path) as data:
                long = data["long"].astype(np.float32) / 65535.0
            linear = libraw_decode(torch.from_numpy(long)).permute(2, 0, 1)
            luminance = (0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]).flatten()
            gain = float((0.18 / luminance.median().clamp_min(1e-4)).clamp(1, 300))
            rgb = srgb_encode((linear * gain).clamp(0, 1)).permute(1, 2, 0).numpy()
            image = Image.fromarray((rgb * 255).round().astype(np.uint8))
            edges = np.asarray(detector(image, detect_resolution=512, image_resolution=512, safe=False))
            if edges.ndim == 3:
                edges = edges[..., 0]
            if edges.shape != long.shape[:2]:
                raise ValueError(f"Outline map {edges.shape} does not match crop {long.shape[:2]}")
            np.save(target, edges.astype(np.uint8))
            if index % 500 == 0:
                print(f"EDGE_PROGRESS split={split} {index}/{len(files)}", flush=True)
        print(f"EDGES_DONE split={split} crops={len(files)}", flush=True)


if __name__ == "__main__":
    main()
