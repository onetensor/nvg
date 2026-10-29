"""Cache aligned multi-frame crops from the SID Sony set.

Every Sony scene was shot on a tripod with several independent short
exposures, so the short frames of one scene at one crop position form a real
static burst. Training uses them to reproduce the runtime's temporal
accumulation with genuinely independent noise.

Each output file holds one crop: ``long`` (uint16 HWC), ``short_00``..
``short_NN`` (uint16 HWC, stored under separate keys so loaders read only the
frames they use) and ``ratios`` (exposure ratio per short frame, capped at 300).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.dataset import Record, discover_records, image_rgb


def quantize(value: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(value, 0, 1) * 65535).astype(np.uint16)


def group_records(records: list[Record]) -> list[tuple[Path, list[Record]]]:
    groups: dict[Path, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.target_path].append(record)
    return [(target, sorted(items, key=lambda item: item.input_path.name)) for target, items in sorted(groups.items())]


def build_group(
    group_index: int,
    target_path: Path,
    records: list[Record],
    destination: Path,
    split: str,
    crops: int,
    patch_size: int,
    max_frames: int,
    seed: int,
) -> int:
    outputs = [destination / f"{group_index:04d}-{crop:02d}.npz" for crop in range(crops)]
    if all(path.exists() and path.stat().st_size > 0 for path in outputs):
        return 0
    rng = random.Random(seed + group_index * 1_000_003)
    target = image_rgb(target_path)
    shorts = [image_rgb(record.input_path) for record in records]
    height = min(target.shape[0], *(short.shape[0] for short in shorts))
    width = min(target.shape[1], *(short.shape[1] for short in shorts))
    for crop, output in enumerate(outputs):
        if output.exists() and output.stat().st_size > 0:
            continue
        if split == "train":
            y = rng.randrange(0, height - patch_size + 1) // 2 * 2
            x = rng.randrange(0, width - patch_size + 1) // 2 * 2
            chosen = rng.sample(range(len(shorts)), min(max_frames, len(shorts)))
        else:
            fraction = (crop + 1) / (crops + 1)
            y = int((height - patch_size) * fraction) // 2 * 2
            x = int((width - patch_size) * (1 - fraction)) // 2 * 2
            chosen = list(range(min(max_frames, len(shorts))))
        arrays = {
            "long": quantize(target[y:y + patch_size, x:x + patch_size]),
            "ratios": np.array([records[index].gain for index in chosen], dtype=np.float32),
        }
        for frame, index in enumerate(chosen):
            arrays[f"short_{frame:02d}"] = quantize(shorts[index][y:y + patch_size, x:x + patch_size])
        temporary = output.with_suffix(".partial.npz")
        np.savez(temporary, **arrays)
        temporary.replace(output)
    return 1


def build_split(args: argparse.Namespace, split: str, crops: int, offset: int) -> dict[str, int | str]:
    groups = group_records(discover_records(Path(args.data), split))
    if split != "train" and args.max_eval_scenes:
        indices = np.linspace(0, len(groups) - 1, min(args.max_eval_scenes, len(groups)), dtype=int)
        groups = [groups[index] for index in indices]
    destination = Path(args.output) / "bursts" / split
    destination.mkdir(parents=True, exist_ok=True)
    completed = 0
    # Processes, not threads: LibRaw decoding holds the GIL often enough that a
    # thread pool saturates only a few cores.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(
            build_group, index, target, records, destination, split,
            crops, args.patch_size, args.max_frames, args.seed + offset,
        ) for index, (target, records) in enumerate(groups)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(groups):
                print(f"BURST_PROGRESS split={split} scenes={completed}/{len(groups)}", flush=True)
    return {"split": split, "scenes": len(groups), "crops": len(groups) * crops,
            "patch_size": args.patch_size, "max_frames": args.max_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache aligned SID short-exposure bursts")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--train-crops", type=int, default=16)
    parser.add_argument("--eval-crops", type=int, default=2)
    parser.add_argument("--max-eval-scenes", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    reports = [
        build_split(args, split, args.train_crops if split == "train" else args.eval_crops, offset)
        for offset, split in enumerate(("train", "val", "test"))
    ]
    manifest = Path(args.output) / "bursts" / "manifest.json"
    manifest.write_text(json.dumps(reports, indent=2) + "\n")
    print("BURST_DATASET_COMPLETE " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
