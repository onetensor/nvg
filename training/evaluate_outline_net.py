"""Outline network vs classic edges on held-out bursts (tolerance boundary F-score).

Every method is scored at its own best threshold against thinned teacher
outlines with a 2-pixel tolerance: the dedicated outline network, Sobel on the
frozen model's denoised output, and Sobel on the noisy gained input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.outline_net import best_f_score, load_outline, outline_inputs, sobel_magnitude, tolerance_counts
from training.phone_sim import BurstDataset, collate_list, degrade_bursts
from training.sid_model import direct_input, load_sid

THRESHOLDS = torch.linspace(0.02, 0.98, 49)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--outline", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    sid = load_sid(args.sid, device).eval()
    outline = load_outline(args.outline, device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with torch.inference_mode():
        for frames in args.frames:
            loader = DataLoader(BurstDataset(args.data, "test", 256, augment=False, eval_frames=frames),
                                batch_size=1, num_workers=4, collate_fn=collate_list)
            counts = {name: torch.zeros(4, len(THRESHOLDS), device=device)
                      for name in ("outline_net", "sobel_output", "sobel_input")}
            for index, samples in enumerate(loader):
                thin: list[torch.Tensor] = []
                inputs, _, gain, frame_counts = degrade_bursts(samples, device, False, args.noise, frames,
                                                               thin_outlines=thin)
                packed = direct_input(sid, inputs, gain[:, None, None, None], frame_counts)
                image = sid(packed)
                features = outline_inputs(packed, image)
                maps = {
                    "outline_net": torch.sigmoid(outline(features)),
                    "sobel_output": features[:, 1:2].clamp(0, 1),
                    "sobel_input": (sobel_magnitude(features[:, 2:3]) / 2).clamp(0, 1),
                }
                reference = torch.stack(thin)
                for name, value in maps.items():
                    counts[name] += tolerance_counts(value, reference, THRESHOLDS.to(device))
                if index < args.images:
                    tiles = [features[0, 2], image[0, 0].clamp(0, 1), maps["sobel_output"][0, 0],
                             maps["outline_net"][0, 0], reference[0, 0]]
                    row = torch.cat(tiles, dim=-1).cpu().numpy()
                    Image.fromarray((row * 255).round().clip(0, 255).astype(np.uint8)).save(output / f"f{frames}-{index:02d}.png")
            condition = {name: best_f_score(value.cpu(), THRESHOLDS) for name, value in counts.items()}
            summary[f"f{frames}_n{args.noise:g}"] = condition
            print(f"f{frames}_n{args.noise:g} " + "  ".join(
                f"{name}: F {v['f']:.3f} (P {v['precision']:.2f} R {v['recall']:.2f} @ {v['threshold']:.2f})"
                for name, v in condition.items()), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "metric": "boundary F-score, 2 px tolerance, thinned HED outlines, best threshold per method",
        "columns": ["noisy input (luma)", "denoised", "sobel on denoised", "outline network", "teacher (thinned)"],
        "conditions": summary,
    }, indent=2) + "\n")
    print("OUTLINE_NET_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
