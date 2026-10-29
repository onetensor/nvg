"""Scores object outlines on simulated dark captures against teacher outlines.

The reference is the HED outline map of the clean, brightened long exposure
(build_edge_targets). Each method's map is binarized at its own best threshold
over the whole test set (the boundary-detection "ODS" F-score), which favours
every method equally:

* learned: the model's outline plane (sigmoid of its logits)
* sobel_output: Sobel magnitude of the model's denoised luminance
* sobel_input: Sobel magnitude of the noisy input luminance after gain
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

from training.phone_sim import BurstDataset, collate_list, degrade_bursts, srgb_encode
from training.sid_model import direct_input, load_sid, luma

THRESHOLDS = torch.linspace(0.02, 0.98, 49)


def sobel(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device)
    kernels = torch.stack((kernel_x, kernel_x.T))[:, None]
    gradients = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernels)
    return gradients.square().sum(dim=1, keepdim=True).sqrt()


def main() -> None:
    parser = argparse.ArgumentParser(description="Outline F-scores on simulated dark captures")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="Checkpoint with a luminance + outline output")
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_sid(args.model, device).eval()
    if model.out_channels != 8:
        raise ValueError("The model has no outline plane")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    methods = ("learned", "sobel_output", "sobel_input")
    summary = {}
    with torch.inference_mode():
        for frames in args.frames:
            dataset = BurstDataset(args.data, "test", args.output_size, augment=False, eval_frames=frames)
            loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_list)
            # Per threshold: true positives, predicted positives, actual positives.
            counts = {method: torch.zeros(3, len(THRESHOLDS)) for method in methods}
            for index, batch in enumerate(loader):
                outlines: list[torch.Tensor] = []
                inputs, _, gain, frame_counts = degrade_bursts(batch, device, False, args.noise, frames, outlines=outlines)
                gain = gain[:, None, None, None]
                result = model(direct_input(model, inputs, gain, frame_counts))
                image = result[:, :1].clamp(0, 1)
                noisy = luma(srgb_encode((inputs * gain).clamp(0, 1)))
                maps = {
                    "learned": torch.sigmoid(result[:, 1:2]),
                    # Sobel on [0, 1] images peaks near 4; scale into [0, 1] for thresholds.
                    "sobel_output": (sobel(image) / 2).clamp(0, 1),
                    "sobel_input": (sobel(noisy) / 2).clamp(0, 1),
                }
                reference = torch.stack(outlines) > 0.5
                for method, value in maps.items():
                    predicted = value[None] > THRESHOLDS.to(device)[:, None, None, None, None]
                    counts[method][0] += (predicted & reference).flatten(1).sum(1).float().cpu()
                    counts[method][1] += predicted.flatten(1).sum(1).float().cpu()
                    counts[method][2] += reference.sum().float().cpu()
                if index < args.images:
                    tiles = [noisy[0, 0], image[0, 0], maps["sobel_output"][0, 0], maps["learned"][0, 0],
                             torch.stack(outlines)[0, 0]]
                    row = torch.cat(tiles, dim=-1).cpu().numpy()
                    Image.fromarray((row * 255).round().clip(0, 255).astype(np.uint8)).save(output / f"f{frames}-{index:02d}.png")
            condition = {}
            for method, (true_positive, predicted, actual) in counts.items():
                precision = true_positive / predicted.clamp_min(1)
                recall = true_positive / actual.clamp_min(1)
                f_score = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
                best = int(f_score.argmax())
                condition[method] = {"f": float(f_score[best]), "precision": float(precision[best]),
                                     "recall": float(recall[best]), "threshold": float(THRESHOLDS[best])}
            summary[f"f{frames}_n{args.noise:g}"] = condition
            print(f"f{frames}_n{args.noise:g} " + "  ".join(
                f"{method}: F {value['f']:.3f} (P {value['precision']:.2f} R {value['recall']:.2f} @ {value['threshold']:.2f})"
                for method, value in condition.items()), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "columns": ["noisy input", "model image", "sobel on output", "learned outlines", "teacher outlines"],
        "conditions": summary,
    }, indent=2) + "\n")
    print("OUTLINE_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
