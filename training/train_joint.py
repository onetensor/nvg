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

from training.adapter_model import PseudoRawAdapter
from training.dataset import AdapterDataset
from training.losses import charbonnier, psnr, ssim
from training.sid_model import load_sid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly fine-tune a warm-started pseudo-RAW adapter and pretrained SID"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--adapter-learning-rate", type=float, default=3e-6)
    parser.add_argument("--sid-learning-rate", type=float, default=3e-7)
    parser.add_argument("--raw-weight", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1_000)
    return parser.parse_args()


def load_adapter(path: str, device: torch.device) -> tuple[PseudoRawAdapter, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    adapter = PseudoRawAdapter(
        width=int(payload.get("width", 64)),
        residual_scale=float(payload.get("residual_scale", 0.10)),
    ).to(device)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter, payload


@torch.no_grad()
def validate(
    adapter: PseudoRawAdapter,
    sid: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    sid.eval()
    result = {"learned_psnr": 0.0, "learned_ssim": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= 12:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        gain = batch["gain"].to(device, non_blocking=True)[:, None, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = sid((adapter(inputs) * gain).clamp(0, 1)).float().clamp(0, 1)
        result["learned_psnr"] += psnr(prediction, targets).item()
        result["learned_ssim"] += ssim(prediction, targets).item()
        samples += 1
    adapter.train()
    sid.train()
    return {key: value / max(samples, 1) for key, value in result.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for joint fine-tuning")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    adapter, source_adapter = load_adapter(args.adapter, device)
    sid = load_sid(args.sid, device)
    adapter.train()
    sid.train()

    optimizer = AdamW(
        [
            {"params": adapter.parameters(), "lr": args.adapter_learning_rate},
            {"params": sid.parameters(), "lr": args.sid_learning_rate},
        ],
        betas=(0.9, 0.99),
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=min(args.adapter_learning_rate, args.sid_learning_rate) * 0.02,
    )
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
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 4),
        pin_memory=True,
    )
    iterator = iter(train_loader)
    log_path = output / "metrics.jsonl"
    started = time.time()

    def save_pair(stem: str, step: int) -> None:
        torch.save(
            {
                "state_dict": adapter.state_dict(),
                "width": adapter.width,
                "residual_scale": adapter.residual_scale,
                "parameter_count": adapter.parameter_count,
                "step": step,
                "joint_finetuned": True,
                "source_adapter_step": source_adapter.get("step"),
                "args": vars(args),
            },
            output / f"adapter-{stem}.pt",
        )
        torch.save(
            {
                "state_dict": sid.state_dict(),
                "step": step,
                "joint_finetuned": True,
                "source_sid": args.sid,
                "args": vars(args),
            },
            output / f"sid-{stem}.pt",
        )

    initial = {"step": 0, **validate(adapter, sid, val_loader, device), "best": True}
    best_score = initial["learned_psnr"] + 10.0 * initial["learned_ssim"]
    save_pair("best", 0)
    with log_path.open("w") as stream:
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
            raw_loss = (
                (pseudo_raw[has_raw] - packed_raw[has_raw]).abs().mean()
                if has_raw.any()
                else output_loss.new_zeros(())
            )
            loss = output_loss + args.raw_weight * raw_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(sid.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            record = {
                "step": step,
                "loss": loss.item(),
                "output_loss": output_loss.item(),
                "raw_loss": raw_loss.item(),
                "adapter_learning_rate": optimizer.param_groups[0]["lr"],
                "sid_learning_rate": optimizer.param_groups[1]["lr"],
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
                save_pair("best", step)
                report["best"] = True
            with log_path.open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            print("VALIDATION " + json.dumps(report), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            save_pair("latest", step)

    (output / "TRAINING_COMPLETE").write_text(f"steps={args.steps}\n")
    print(
        f"JOINT_TRAINING_COMPLETE adapter_parameters={adapter.parameter_count} "
        f"sid_parameters={sum(parameter.numel() for parameter in sid.parameters())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
