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
from training.phone_sim import BurstDataset, PhoneSimDataset, collate_list, degrade_batch, degrade_bursts
from training.losses import charbonnier, psnr, ssim
from training.sid_model import SIDUNet, add_noise_level_input, direct_from_bayer, direct_input, load_sid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an adapter-free SID that takes 2x2-packed linear RGB directly"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-sid", help="4-channel SID checkpoint to warm-start a width-32 model")
    parser.add_argument("--init-direct", help="Direct checkpoint to warm-start from (12-channel ones gain a zero-weight noise plane)")
    parser.add_argument("--in-channels", type=int, choices=(12, 13), default=12,
                        help="13 adds a 1/sqrt(frames averaged) noise-level plane")
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--teacher", help="Direct (12-channel) checkpoint to distill from")
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--protocol",
        choices=("burst", "phone", "legacy"),
        default="phone",
        help="burst: simulated iPhone video with runtime accumulation of real SID bursts; "
             "phone: single simulated frames; legacy: dark ISP input with oracle gain",
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
    if args.init_direct:
        source = load_sid(args.init_direct)
        if source.in_channels == 12 and args.in_channels == 13:
            source = add_noise_level_input(source)
        if source.in_channels != args.in_channels or source.width != args.width:
            raise ValueError("--init-direct does not match --in-channels/--width")
        return source.to(device)
    if args.in_channels != 12:
        raise ValueError("A 13-channel model needs --init-direct")
    if args.init_sid:
        source = load_sid(args.init_sid)
        if source.width != args.width:
            raise ValueError(f"--init-sid width {source.width} does not match --width {args.width}")
        return direct_from_bayer(source).to(device)
    return SIDUNet(in_channels=12, width=args.width).to(device)


Prepared = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
Prepare = Callable[..., Prepared]

# Burst validation: a clean single frame and a noisier 4-frame accumulation.
BURST_VALIDATION = ({"name": "f1", "frames": 1, "extra_noise": 0.0}, {"name": "f4n6", "frames": 4, "extra_noise": 6.0})


def batch_preparer(args: argparse.Namespace, device: torch.device) -> Prepare:
    """(inputs, targets, gain[B,1,1,1], frame counts or None) for a loader batch."""
    def prepare(batch, augment: bool, extra_noise: float | None = None) -> Prepared:
        counts = None
        if args.protocol == "burst":
            inputs, targets, gain, counts = degrade_bursts(batch, device, augment, extra_noise)
        elif args.protocol == "phone":
            inputs, targets, gain = degrade_batch(batch, device, args.output_size, augment)
        else:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            gain = batch["gain"].to(device, non_blocking=True)
        return inputs, targets, gain[:, None, None, None], counts
    return prepare


@torch.no_grad()
def validate(
    model: SIDUNet,
    loader: DataLoader,
    device: torch.device,
    prepare: Prepare,
    extra_noise: float | None = None,
) -> dict[str, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    samples = 0
    for batch in loader:
        inputs, targets, gain, counts = prepare(batch, False, extra_noise)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(direct_input(model, inputs, gain, counts)).float().clamp(0, 1)
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
        if teacher.in_channels not in (12, 13):
            raise ValueError("The teacher must be a direct model")
        teacher = teacher.to(memory_format=torch.channels_last)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-5)

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    perceptual = None
    if args.lpips_weight > 0:
        import lpips
        perceptual = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    collate = None
    val_loaders: list[tuple[str, DataLoader, float | None]] = []
    if args.protocol == "burst":
        train_data = BurstDataset(args.data, "train", args.output_size, augment=True)
        collate = collate_list
    elif args.protocol == "phone":
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
        collate_fn=collate,
    )

    def evaluation_loader(dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=min(args.workers, 8),
                          pin_memory=True, worker_init_fn=worker_init, collate_fn=collate)

    if args.protocol == "burst":
        for config in BURST_VALIDATION:
            dataset = BurstDataset(args.data, "val", args.output_size, augment=False, eval_frames=config["frames"])
            val_loaders.append((config["name"], evaluation_loader(dataset), config["extra_noise"]))
    else:
        val_loaders.append(("val", evaluation_loader(val_data), None))

    def validate_all() -> dict[str, float]:
        reports = {name: validate(model, loader, device, prepare, noise) for name, loader, noise in val_loaders}
        if len(reports) == 1:
            return next(iter(reports.values()))
        merged: dict[str, float] = {
            "psnr": sum(report["psnr"] for report in reports.values()) / len(reports),
            "ssim": sum(report["ssim"] for report in reports.values()) / len(reports),
        }
        for name, report in reports.items():
            merged[f"{name}_psnr"] = report["psnr"]
            merged[f"{name}_ssim"] = report["ssim"]
        return merged
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    prepare = batch_preparer(args, device)
    log_path = output / "metrics.jsonl"
    started = time.time()

    def save(stem: str, step: int, metrics: dict[str, float] | None = None) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": model.in_channels,
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

    initial = {"step": 0, **validate_all(), "best": True}
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
        inputs, targets, gain, counts = prepare(batch, True)
        packed = direct_input(model, inputs, gain, counts).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(packed).float()
            target_loss = charbonnier(prediction, targets) + 0.2 * (1 - ssim(prediction, targets))
            if teacher is not None:
                with torch.no_grad():
                    teacher_input = direct_input(teacher, inputs, gain, counts).contiguous(
                        memory_format=torch.channels_last)
                    reference = teacher(teacher_input).float().clamp(0, 1)
                teacher_loss = charbonnier(prediction, reference)
            else:
                teacher_loss = target_loss.new_zeros(())
        perceptual_loss = (
            perceptual(prediction.clamp(0, 1) * 2 - 1, targets * 2 - 1).mean()
            if perceptual is not None else target_loss.new_zeros(())
        )
        loss = target_loss + args.teacher_weight * teacher_loss + args.lpips_weight * perceptual_loss
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
                "lpips_loss": perceptual_loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - started, 1),
                "steps_per_s": round(step / (time.time() - started), 2),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % args.validate_every == 0 or step == args.steps:
            report = {"step": step, **validate_all()}
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
