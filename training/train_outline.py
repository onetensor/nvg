"""Trains the dedicated outline network on top of a frozen direct SID.

Targets are thinned teacher outlines (see build_edge_targets) widened to a
narrow band, with class-balanced BCE, so the network learns crisp lines.
Validation reports the tolerance-based boundary F-score at the best threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training.outline_net import OutlineNet, best_f_score, outline_inputs, tolerance_counts
from training.phone_sim import BurstDataset, collate_list, degrade_bursts
from training.sid_model import balanced_outline_loss, direct_input, load_sid

THRESHOLDS = torch.linspace(0.05, 0.95, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--sid", required=True, help="Frozen 13-channel mono direct model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--widths", type=int, nargs=3, default=(16, 24, 32))
    parser.add_argument("--band", type=int, default=1, help="Target band half-width in pixels")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--validate-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sid = load_sid(args.sid, device).eval().requires_grad_(False)
    if sid.in_channels != 13 or sid.out_channels != 4:
        raise ValueError("Expected a 13-channel mono direct model")
    model = OutlineNet(tuple(args.widths)).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lambda step: (step + 1) / args.warmup_steps if step < args.warmup_steps else
                         0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup_steps) /
                                                                              max(1, args.steps - args.warmup_steps)))))
    train_loader = DataLoader(BurstDataset(args.data, "train", args.output_size, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True, worker_init_fn=worker_init,
                              prefetch_factor=4, collate_fn=collate_list)
    val_loaders = {frames: DataLoader(BurstDataset(args.data, "val", args.output_size, augment=False, eval_frames=frames),
                                      batch_size=4, num_workers=4, worker_init_fn=worker_init, collate_fn=collate_list)
                   for frames in (1, 4)}

    def batch(samples, augment: bool, noise: float | None = None):
        thin: list[torch.Tensor] = []
        inputs, _, gain, counts = degrade_bursts(samples, device, augment, noise, thin_outlines=thin)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            packed = direct_input(sid, inputs, gain[:, None, None, None], counts)
            image = sid(packed).float()
        return outline_inputs(packed.float(), image), torch.stack(thin)

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        report: dict[str, float] = {}
        for frames, loader in val_loaders.items():
            counts = torch.zeros(4, len(THRESHOLDS), device=device)
            for samples in loader:
                features, thin = batch(samples, False, 2.0 if frames == 4 else 0.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probability = torch.sigmoid(model(features).float())
                counts += tolerance_counts(probability, thin, THRESHOLDS.to(device))
            score = best_f_score(counts.cpu(), THRESHOLDS)
            report[f"f{frames}_f"] = score["f"]
            report[f"f{frames}_threshold"] = score["threshold"]
        report["f"] = (report["f1_f"] + report["f4_f"]) / 2
        model.train()
        return report

    def save(stem: str, step: int, metrics: dict[str, float]) -> None:
        torch.save({"state_dict": model.state_dict(), "widths": list(args.widths), "parameter_count": parameters,
                    "step": step, "metrics": metrics, "sid": args.sid, "args": vars(args)},
                   output / f"outline-{stem}.pt")

    log = (output / "metrics.jsonl").open("w")
    best = -1.0
    started = time.time()
    iterator = iter(train_loader)
    for step in range(0, args.steps + 1):
        if step % args.validate_every == 0 or step == args.steps:
            report = {"step": step, **validate()}
            if report["f"] > best:
                best = report["f"]
                save("best", step, report)
                report["best"] = True
            log.write(json.dumps(report) + "\n")
            log.flush()
            print("VALIDATION " + json.dumps(report), flush=True)
        if step == args.steps:
            break
        try:
            samples = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            samples = next(iterator)
        features, thin = batch(samples, True)
        # A narrow band around the one-pixel outline tolerates slight offsets.
        band = F.max_pool2d(thin, 2 * args.band + 1, stride=1, padding=args.band)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features).float()
        loss = balanced_outline_loss(logits, band)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            record = {"step": step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"],
                      "steps_per_s": round((step + 1) / (time.time() - started), 2)}
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    save("latest", args.steps, {})
    (output / "TRAINING_COMPLETE").write_text(f"steps={args.steps}\n")
    print(f"OUTLINE_TRAINING_COMPLETE parameters={parameters}", flush=True)


if __name__ == "__main__":
    main()
