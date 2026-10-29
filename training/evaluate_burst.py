from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.losses import psnr, ssim
from training.phone_sim import BurstDataset, collate_list, degrade_bursts, srgb_encode
from training.sid_model import direct_input, load_sid

# (frames averaged, extra-noise median photons; 0 = Sony noise only)
CONDITIONS = ((1, 0.0), (1, 6.0), (4, 6.0), (8, 6.0), (1, 2.0), (8, 2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct models on simulated, accumulated iPhone bursts")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--images", type=int, default=6, help="Comparison images saved per condition")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import lpips
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    models = {}
    for entry in args.models:
        name, path = entry.split("=", 1)
        models[name] = load_sid(path, device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    with torch.inference_mode():
        for frames, noise in CONDITIONS:
            condition = f"f{frames}_n{noise:g}"
            dataset = BurstDataset(args.data, args.split, args.output_size, augment=False,
                                   eval_downsample=args.downsample, eval_frames=frames)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_list)
            totals = {name: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0} for name in models}
            for index, batch in enumerate(loader):
                inputs, targets, gain, counts = degrade_bursts(batch, device, False, noise, frames)
                gain = gain[:, None, None, None]
                tiles = [srgb_encode((inputs * gain).clamp(0, 1))]
                for name, model in models.items():
                    prediction = model(direct_input(model, inputs, gain, counts)).clamp(0, 1)
                    totals[name]["psnr"] += psnr(prediction, targets).item()
                    totals[name]["ssim"] += ssim(prediction, targets).item()
                    totals[name]["lpips"] += perceptual(prediction * 2 - 1, targets * 2 - 1).mean().item()
                    tiles.append(prediction)
                tiles.append(targets)
                if index < args.images:
                    row = torch.cat([tile[0] for tile in tiles], dim=-1)
                    array = (row.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(array).save(output / f"{condition}-{index:02d}.png")
            samples = len(dataset)
            summary[condition] = {
                name: {metric: value / samples for metric, value in metrics.items()}
                for name, metrics in totals.items()
            }
            print(f"{condition:9s} " + "  ".join(
                f"{name}: {m['psnr']:.2f} dB / {m['ssim']:.3f} / lpips {m['lpips']:.3f}"
                for name, m in summary[condition].items()), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "split": args.split, "downsample": args.downsample, "columns": ["input x gain", *models, "target"],
        "conditions": summary,
    }, indent=2) + "\n")
    print("BURST_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
