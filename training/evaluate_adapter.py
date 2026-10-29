from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.adapter_model import PseudoRawAdapter, manual_pseudo_bayer
from training.dataset import AdapterDataset
from training.losses import psnr, ssim
from training.sid_model import load_sid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--min-psnr-improvement", type=float, default=0.1)
    parser.add_argument("--min-ssim-improvement", type=float, default=0.002)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sid = load_sid(args.sid, device).eval().requires_grad_(False)
    payload = torch.load(args.adapter, map_location=device, weights_only=False)
    adapter = PseudoRawAdapter(
        width=int(payload.get("width", 32)),
        residual_scale=float(payload.get("residual_scale", 0.25)),
    ).to(device).eval()
    adapter.load_state_dict(payload["state_dict"])
    data = AdapterDataset(args.data, "test", args.patch_size, augment=False)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2)
    rows: list[dict[str, float | int]] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.limit and index >= args.limit:
                break
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            packed_raw = batch["raw"].to(device)
            has_raw = bool(batch["has_raw"].all())
            gain = batch["gain"].to(device)[:, None, None, None]
            learned = sid((adapter(inputs) * gain).clamp(0, 1)).clamp(0, 1)
            manual = sid((manual_pseudo_bayer(inputs) * gain).clamp(0, 1)).clamp(0, 1)
            row = {
                "index": index,
                "learned_psnr": psnr(learned, targets).item(),
                "learned_ssim": ssim(learned, targets).item(),
                "manual_psnr": psnr(manual, targets).item(),
                "manual_ssim": ssim(manual, targets).item(),
            }
            if has_raw:
                true_raw = sid((packed_raw * gain).clamp(0, 1)).clamp(0, 1)
                row["true_raw_psnr"] = psnr(true_raw, targets).item()
                row["true_raw_ssim"] = ssim(true_raw, targets).item()
            rows.append(row)
    summary: dict[str, float | int | bool] = {
        "samples": len(rows),
        **{key: sum(float(row[key]) for row in rows) / max(len(rows), 1)
           for key in ("learned_psnr", "learned_ssim", "manual_psnr", "manual_ssim")},
    }
    summary["psnr_improvement"] = float(summary["learned_psnr"]) - float(summary["manual_psnr"])
    summary["ssim_improvement"] = float(summary["learned_ssim"]) - float(summary["manual_ssim"])
    raw_rows = [row for row in rows if "true_raw_psnr" in row]
    if raw_rows:
        summary["true_raw_samples"] = len(raw_rows)
        summary["true_raw_psnr"] = sum(float(row["true_raw_psnr"]) for row in raw_rows) / len(raw_rows)
        summary["true_raw_ssim"] = sum(float(row["true_raw_ssim"]) for row in raw_rows) / len(raw_rows)
    summary["accepted"] = (
        float(summary["psnr_improvement"]) >= args.min_psnr_improvement
        or float(summary["ssim_improvement"]) >= args.min_ssim_improvement
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.require_improvement and not summary["accepted"]:
        raise SystemExit("Learned adapter did not beat the deterministic baseline; refusing to package it.")


if __name__ == "__main__":
    main()
