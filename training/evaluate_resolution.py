"""Compares models at the resolution each one would run at, scored at the higher one.

The app can run a wide model at 180p or a slim one at 360p for similar cost.
Each test burst is simulated twice from the same crop and seed: once with 4x
runtime downsampling (180p-like, output upscaled 2x for display, as the
browser would) and once with 2x (360p-like). Both are scored in luma against
the 2x clean target, so finer detail at 360p counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from training.losses import psnr, ssim
from training.phone_sim import BurstDataset, collate_list, degrade_bursts
from training.sid_model import direct_input, load_sid, luma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--low", nargs="+", required=True, help="name=checkpoint run at 4x downsampling (180p-like)")
    parser.add_argument("--high", nargs="+", required=True, help="name=checkpoint run at 2x downsampling (360p-like)")
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import lpips
    device = torch.device(args.device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    load = lambda entries: {name: load_sid(path, device).eval()
                            for name, path in (entry.split("=", 1) for entry in entries)}
    low, high = load(args.low), load(args.high)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with torch.inference_mode():
        for frames in args.frames:
            datasets = {
                "low": BurstDataset(args.data, "test", 128, augment=False, eval_downsample=4, eval_frames=frames),
                "high": BurstDataset(args.data, "test", 256, augment=False, eval_downsample=2, eval_frames=frames),
            }
            loaders = {key: DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_list)
                       for key, dataset in datasets.items()}
            totals = {name: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0} for name in (*low, *high)}
            for index, (low_batch, high_batch) in enumerate(zip(loaders["low"], loaders["high"])):
                low_in, _, low_gain, low_counts = degrade_bursts(low_batch, device, False, args.noise, frames)
                high_in, target, high_gain, high_counts = degrade_bursts(high_batch, device, False, args.noise, frames)
                target = luma(target)
                outputs = {}
                for name, model in low.items():
                    prediction = model(direct_input(model, low_in, low_gain[:, None, None, None], low_counts))
                    prediction = luma(prediction) if prediction.shape[1] == 3 else prediction[:, :1]
                    outputs[name] = F.interpolate(prediction, scale_factor=2, mode="bilinear", align_corners=False)
                for name, model in high.items():
                    prediction = model(direct_input(model, high_in, high_gain[:, None, None, None], high_counts))
                    outputs[name] = luma(prediction) if prediction.shape[1] == 3 else prediction[:, :1]
                for name, prediction in outputs.items():
                    prediction = prediction.clamp(0, 1)
                    totals[name]["psnr"] += psnr(prediction, target).item()
                    totals[name]["ssim"] += ssim(prediction, target).item()
                    totals[name]["lpips"] += perceptual((prediction * 2 - 1).expand(-1, 3, -1, -1),
                                                        (target * 2 - 1).expand(-1, 3, -1, -1)).mean().item()
                if index < args.images:
                    row = torch.cat([outputs[name].clamp(0, 1)[0] for name in outputs] + [target[0]], dim=-1)
                    array = (row[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(array).save(output / f"f{frames}-{index:02d}.png")
            samples = len(datasets["high"])
            summary[f"f{frames}_n{args.noise:g}"] = {
                name: {metric: value / samples for metric, value in metrics.items()} for name, metrics in totals.items()
            }
            print(f"f{frames}_n{args.noise:g} " + "  ".join(
                f"{name}: {m['psnr']:.2f} / {m['ssim']:.3f} / lpips {m['lpips']:.3f}"
                for name, m in summary[f"f{frames}_n{args.noise:g}"].items()), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "columns": [*low, *high, "target"], "low_resolution": list(low), "high_resolution": list(high),
        "conditions": summary,
    }, indent=2) + "\n")
    print("RESOLUTION_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
