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
from training.sid_model import direct_input, load_sid, luma


def color_fit_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR after the best global affine color map (3x3 + offset) from prediction to target.

    The gap to plain PSNR measures how much error is global color/tone rather
    than noise or missing detail.
    """
    x = prediction[0].flatten(1).T
    y = target[0].flatten(1).T
    # A luma fit is gain + offset; an RGB fit is a 3x3 matrix + offset.
    design = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    weights = torch.linalg.lstsq(design.double(), y.double()).solution.float()
    fitted = (design @ weights).T.reshape(target.shape).clamp(0, 1)
    return psnr(fitted, target).item()


# (frames averaged, extra-noise median photons; 0 = Sony noise only)
CONDITIONS = ((1, 0.0), (1, 6.0), (4, 6.0), (8, 6.0), (1, 2.0), (8, 2.0), (32, 2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct models on simulated, accumulated iPhone bursts")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--images", type=int, default=6, help="Comparison images saved per condition")
    parser.add_argument("--target", choices=("phone", "sid"), default="phone")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="*", help="Subset such as f1_n2 f8_n2 (default: all)")
    parser.add_argument("--luma", action="store_true",
                        help="Score everything in luma (required when any model is mono)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import lpips
    device = torch.device(args.device)
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
            if args.conditions and condition not in args.conditions:
                continue
            dataset = BurstDataset(args.data, args.split, args.output_size, augment=False,
                                   eval_downsample=args.downsample, eval_frames=frames)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_list)
            totals = {name: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "psnr_color_fit": 0.0} for name in models}
            for index, batch in enumerate(loader):
                inputs, targets, gain, counts = degrade_bursts(batch, device, False, noise, frames, target=args.target)
                gain = gain[:, None, None, None]
                shown = srgb_encode((inputs * gain).clamp(0, 1))
                if args.luma:
                    targets = luma(targets)
                tiles = [luma(shown) if args.luma else shown]
                for name, model in models.items():
                    prediction = model(direct_input(model, inputs, gain, counts)).clamp(0, 1)
                    if prediction.shape[1] == 2:  # luma + outline logits
                        prediction = prediction[:, :1]
                    if args.luma and prediction.shape[1] == 3:
                        prediction = luma(prediction)
                    if prediction.shape[1] != targets.shape[1]:
                        raise ValueError(f"{name} is mono; pass --luma")
                    totals[name]["psnr"] += psnr(prediction, targets).item()
                    totals[name]["ssim"] += ssim(prediction, targets).item()
                    totals[name]["lpips"] += perceptual((prediction * 2 - 1).expand(-1, 3, -1, -1),
                                                        (targets * 2 - 1).expand(-1, 3, -1, -1)).mean().item()
                    totals[name]["psnr_color_fit"] += color_fit_psnr(prediction, targets)
                    tiles.append(prediction)
                tiles.append(targets)
                if index < args.images:
                    row = torch.cat([tile[0].expand(3, -1, -1) for tile in tiles], dim=-1)
                    array = (row.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(array).save(output / f"{condition}-{index:02d}.png")
            samples = len(dataset)
            summary[condition] = {
                name: {metric: value / samples for metric, value in metrics.items()}
                for name, metrics in totals.items()
            }
            print(f"{condition:9s} " + "  ".join(
                f"{name}: {m['psnr']:.2f} dB / {m['ssim']:.3f} / lpips {m['lpips']:.3f} / fit {m['psnr_color_fit']:.2f}"
                for name, m in summary[condition].items()), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "split": args.split, "downsample": args.downsample, "target": args.target, "luma": args.luma, "columns": ["input x gain", *models, "target"],
        "conditions": summary,
    }, indent=2) + "\n")
    print("BURST_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
