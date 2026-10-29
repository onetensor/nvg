from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training.dataset import AdapterDataset
from training.phone_sim import PhoneSimDataset, degrade_batch
from training.losses import charbonnier, psnr, ssim
from training.sid_model import SIDUNet, direct_from_bayer, load_sid, packed_linear_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an adapter-free SID that takes 2x2-packed linear RGB directly"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-sid", help="4-channel SID checkpoint to warm-start a width-32 model")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--teacher", help="Direct (12-channel) checkpoint to distill from")
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--protocol",
        choices=("phone", "legacy"),
        default="phone",
        help="phone: simulated iPhone video with runtime auto gain; legacy: dark ISP input with oracle gain",
    )
    parser.add_argument("--output-size", type=int, default=256, help="Model input size for --protocol phone")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validate-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=5_000)
    return parser.parse_args()


def worker_init(_: int) -> None:
    # The camera-pipeline augmentation runs torch ops per sample; one thread
    # per worker avoids oversubscribing the pod's cores.
    torch.set_num_threads(1)


def build_model(args: argparse.Namespace, device: torch.device) -> SIDUNet:
    if args.init_sid:
        source = load_sid(args.init_sid)
        if source.width != args.width:
            raise ValueError(f"--init-sid width {source.width} does not match --width {args.width}")
        return direct_from_bayer(source).to(device)
    return SIDUNet(in_channels=12, width=args.width).to(device)


Prepare = Callable[[dict[str, torch.Tensor], bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def batch_preparer(args: argparse.Namespace, device: torch.device) -> Prepare:
    """(inputs, targets, gain[B,1,1,1]) for a loader batch; the phone protocol degrades on the GPU."""
    def prepare(batch: dict[str, torch.Tensor], augment: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if args.protocol == "phone":
            inputs, targets, gain = degrade_batch(batch, device, args.output_size, augment)
        else:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            gain = batch["gain"].to(device, non_blocking=True)
        return inputs, targets, gain[:, None, None, None]
    return prepare


@torch.no_grad()
def validate(model: SIDUNet, loader: DataLoader, device: torch.device, prepare: Prepare) -> dict[str, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    samples = 0
    for batch in loader:
        inputs, targets, gain = prepare(batch, False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(packed_linear_rgb(inputs, gain)).float().clamp(0, 1)
        for index in range(prediction.shape[0]):
            total_psnr += psnr(prediction[index:index + 1], targets[index:index + 1]).item()
            total_ssim += ssim(prediction[index:index + 1], targets[index:index + 1]).item()
            samples += 1
    model.train()
    return {"psnr": total_psnr / max(samples, 1), "ssim": total_ssim / max(samples, 1), "samples": samples}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # CPU is only for smoke tests of the training loop.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device).to(memory_format=torch.channels_last)
    model.train()
    teacher = None
    if args.teacher:
        teacher = load_sid(args.teacher, device).eval().requires_grad_(False)
        if teacher.in_channels != 12:
            raise ValueError("The teacher must be a direct 12-channel model")
        teacher = teacher.to(memory_format=torch.channels_last)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-5)

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    if args.protocol == "phone":
        train_data = PhoneSimDataset(args.data, "train", augment=True)
        val_data = PhoneSimDataset(args.data, "val", augment=False)
    else:
        train_data = AdapterDataset(args.data, "train", args.patch_size, augment=True)
        val_data = AdapterDataset(args.data, "val", args.patch_size, augment=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
        worker_init_fn=worker_init,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        num_workers=min(args.workers, 8),
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    prepare = batch_preparer(args, device)
    log_path = output / "metrics.jsonl"
    started = time.time()

    def save(stem: str, step: int, metrics: dict[str, float] | None = None) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": 12,
                "width": args.width,
                "input": "packed_linear_rgb",
                "protocol": args.protocol,
                "parameter_count": parameter_count,
                "step": step,
                "metrics": metrics,
                "args": vars(args),
            },
            output / f"direct-{stem}.pt",
        )

    initial = {"step": 0, **validate(model, val_loader, device, prepare), "best": True}
    best_score = initial["psnr"] + 10.0 * initial["ssim"]
    save("best", 0, initial)
    with log_path.open("w") as stream:
        stream.write(json.dumps({"parameters": parameter_count, **initial}) + "\n")
    print("VALIDATION " + json.dumps(initial), flush=True)

    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        inputs, targets, gain = prepare(batch, True)
        packed = packed_linear_rgb(inputs, gain).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(packed).float()
            target_loss = charbonnier(prediction, targets) + 0.2 * (1 - ssim(prediction, targets))
            if teacher is not None:
                with torch.no_grad():
                    reference = teacher(packed).float().clamp(0, 1)
                teacher_loss = charbonnier(prediction, reference)
            else:
                teacher_loss = target_loss.new_zeros(())
            loss = target_loss + args.teacher_weight * teacher_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 50 == 0:
            record = {
                "step": step,
                "loss": loss.item(),
                "target_loss": target_loss.item(),
                "teacher_loss": teacher_loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - started, 1),
                "steps_per_s": round(step / (time.time() - started), 2),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % args.validate_every == 0 or step == args.steps:
            report = {"step": step, **validate(model, val_loader, device, prepare)}
            score = report["psnr"] + 10.0 * report["ssim"]
            if score > best_score:
                best_score = score
                save("best", step, report)
                report["best"] = True
            with log_path.open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            print("VALIDATION " + json.dumps(report), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            save("latest", step)

    (output / "TRAINING_COMPLETE").write_text(f"steps={args.steps}\n")
    print(f"DIRECT_TRAINING_COMPLETE width={args.width} parameters={parameter_count}", flush=True)


if __name__ == "__main__":
    main()
