from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import AdapterDataset
from training.losses import psnr, ssim
from training.phone_sim import PhoneSimDataset, degrade_batch
from training.sid_model import load_sid, packed_linear_rgb
from training.train_joint import load_adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct SID checkpoints with the deployed adapter + SID pair on held-out patches"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-sid", required=True)
    parser.add_argument("--baseline-adapter", required=True)
    parser.add_argument("--direct", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--protocol", choices=("phone", "legacy"), default="phone")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=2, help="Runtime area downsampling for --protocol phone")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_sid = load_sid(args.baseline_sid, device).eval()
    baseline_adapter, _ = load_adapter(args.baseline_adapter, device)
    baseline_adapter.eval()
    direct = {}
    for entry in args.direct:
        name, path = entry.split("=", 1)
        direct[name] = load_sid(path, device).eval()

    dataset = (
        PhoneSimDataset(args.data, args.split, augment=False)
        if args.protocol == "phone"
        else AdapterDataset(args.data, args.split, args.patch_size, augment=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    names = ["baseline_joint", *direct]
    totals = {name: {"psnr": 0.0, "ssim": 0.0} for name in names}
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.protocol == "phone":
                inputs, targets, gain = degrade_batch(batch, device, args.output_size, False, args.downsample)
                gain = gain[:, None, None, None]
            else:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                gain = batch["gain"].to(device)[:, None, None, None]
            predictions = {
                "baseline_joint": baseline_sid((baseline_adapter(inputs) * gain).clamp(0, 1)),
                **{name: model(packed_linear_rgb(inputs, gain)) for name, model in direct.items()},
            }
            row: dict[str, float | int] = {"index": index}
            for name, prediction in predictions.items():
                prediction = prediction.clamp(0, 1)
                row[f"{name}_psnr"] = psnr(prediction, targets).item()
                row[f"{name}_ssim"] = ssim(prediction, targets).item()
                totals[name]["psnr"] += float(row[f"{name}_psnr"])
                totals[name]["ssim"] += float(row[f"{name}_ssim"])
            rows.append(row)

    samples = len(rows)
    summary = {
        "split": args.split,
        "protocol": args.protocol,
        **({"downsample": args.downsample} if args.protocol == "phone" else {}),
        "samples": samples,
        **{name: {metric: value / max(samples, 1) for metric, value in metrics.items()}
           for name, metrics in totals.items()},
    }
    for name, model in direct.items():
        summary[name]["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        summary[name]["width"] = model.width
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n")
    print("EVALUATION " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
