from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.adapter_model import PseudoRawAdapter, manual_pseudo_bayer
from training.dataset import AdapterDataset
from training.losses import charbonnier, psnr, ssim
from training.sid_model import load_sid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB → pseudo-RAW while keeping SID frozen")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--output", default="training/runs/adapter")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--raw-weight", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


@torch.no_grad()
def validate(adapter: PseudoRawAdapter, sid: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    adapter.eval()
    metrics = {"learned_psnr": 0.0, "learned_ssim": 0.0, "manual_psnr": 0.0, "manual_ssim": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= 12:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        gain = batch["gain"].to(device, non_blocking=True)[:, None, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            learned = sid((adapter(inputs) * gain).clamp(0, 1)).float().clamp(0, 1)
            manual = sid((manual_pseudo_bayer(inputs) * gain).clamp(0, 1)).float().clamp(0, 1)
        metrics["learned_psnr"] += psnr(learned, targets).item()
        metrics["learned_ssim"] += ssim(learned, targets).item()
        metrics["manual_psnr"] += psnr(manual, targets).item()
        metrics["manual_ssim"] += ssim(manual, targets).item()
        samples += 1
    adapter.train()
    return {key: value / max(samples, 1) for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sid = load_sid(args.sid, device).eval()
    sid.requires_grad_(False)
    adapter = PseudoRawAdapter(width=args.width, residual_scale=args.residual_scale).to(device).train()
    optimizer = AdamW(adapter.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.02)
    train_data = AdapterDataset(args.data, "train", args.patch_size, augment=True)
    val_data = AdapterDataset(args.data, "val", args.patch_size, augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)
    iterator = iter(train_loader)
    log_path = output / "metrics.jsonl"
    started = time.time()
    best_score = float("-inf")

    def checkpoint(step: int) -> dict[str, object]:
        return {
            "state_dict": adapter.state_dict(),
            "width": adapter.width,
            "residual_scale": adapter.residual_scale,
            "parameter_count": adapter.parameter_count,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }

    initial = {"step": 0, **validate(adapter, sid, val_loader, device), "best": True}
    best_score = initial["learned_psnr"] + 10.0 * initial["learned_ssim"]
    torch.save(checkpoint(0), output / "adapter-best.pt")
    with log_path.open("a") as stream:
        stream.write(json.dumps(initial) + "\n")
    print("VALIDATION " + json.dumps(initial), flush=True)

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        packed_raw = batch["raw"].to(device, non_blocking=True)
        has_raw = batch["has_raw"].to(device, non_blocking=True)
        gain = batch["gain"].to(device, non_blocking=True)[:, None, None, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pseudo_raw = adapter(inputs)
            prediction = sid((pseudo_raw * gain).clamp(0, 1)).float()
            output_loss = charbonnier(prediction, targets) + 0.2 * (1 - ssim(prediction, targets))
            if has_raw.any():
                raw_loss = (pseudo_raw[has_raw] - packed_raw[has_raw]).abs().mean()
            else:
                raw_loss = output_loss.new_zeros(())
            loss = output_loss + args.raw_weight * raw_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            record = {
                "step": step,
                "loss": loss.item(),
                "output_loss": output_loss.item(),
                "raw_loss": raw_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - started,
                "vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % args.validate_every == 0 or step == args.steps:
            report = {"step": step, **validate(adapter, sid, val_loader, device)}
            score = report["learned_psnr"] + 10.0 * report["learned_ssim"]
            if score > best_score:
                best_score = score
                torch.save(checkpoint(step), output / "adapter-best.pt")
                report["best"] = True
            with log_path.open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            print("VALIDATION " + json.dumps(report), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            torch.save(checkpoint(step), output / "adapter-latest.pt")

    (output / "TRAINING_COMPLETE").write_text(f"steps={args.steps}\n")
    print(f"TRAINING_COMPLETE parameters={adapter.parameter_count}", flush=True)


if __name__ == "__main__":
    main()
