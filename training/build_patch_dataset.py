from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import random
from pathlib import Path

import numpy as np
import rawpy

from training.dataset import Record, discover_records, image_rgb, pack_sony


def quantize(value: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(value, 0, 1) * 65535).astype(np.uint16)


@functools.lru_cache(maxsize=8)
def target_image(path: str) -> np.ndarray:
    return image_rgb(Path(path))


def valid_patch(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return {"input", "target", "raw", "gain", "has_raw"}.issubset(data.files)
    except (OSError, ValueError):
        return False


def build_record(
    record_index: int,
    record: Record,
    destination: Path,
    split: str,
    patches_per_record: int,
    patch_size: int,
    seed: int,
) -> int:
    outputs = [destination / f"{record_index:05d}-{patch_index:02d}.npz"
               for patch_index in range(patches_per_record)]
    if all(valid_patch(path) for path in outputs):
        return 0
    rng = random.Random(seed + record_index * 1_000_003)
    input_rgb = image_rgb(record.input_path)
    target_rgb = target_image(str(record.target_path))
    if record.raw_path:
        with rawpy.imread(str(record.raw_path)) as raw:
            packed_raw = pack_sony(raw)
    else:
        packed_raw = None
    height = min(input_rgb.shape[0], target_rgb.shape[0])
    width = min(input_rgb.shape[1], target_rgb.shape[1])
    size = min(patch_size, height - height % 32, width - width % 32)
    for patch_index, output in enumerate(outputs):
        if valid_patch(output):
            continue
        if split == "train":
            y = rng.randrange(0, height - size + 1) // 2 * 2
            x = rng.randrange(0, width - size + 1) // 2 * 2
        else:
            fraction = (patch_index + 1) / (patches_per_record + 1)
            y = int((height - size) * fraction) // 2 * 2
            x = int((width - size) * (1 - fraction)) // 2 * 2
        raw_crop = (packed_raw[:, y // 2:(y + size) // 2, x // 2:(x + size) // 2]
                    if packed_raw is not None else np.zeros((4, size // 2, size // 2), np.float32))
        temporary = output.with_suffix(".partial.npz")
        np.savez(
            temporary,
            input=quantize(input_rgb[y:y + size, x:x + size]),
            target=quantize(target_rgb[y:y + size, x:x + size]),
            raw=quantize(raw_crop),
            gain=np.float32(record.gain),
            has_raw=np.bool_(packed_raw is not None),
        )
        temporary.replace(output)
    return 1


def build_split(
    data_root: Path,
    output_root: Path,
    split: str,
    patches_per_record: int,
    patch_size: int,
    max_records: int,
    seed: int,
    workers: int,
) -> dict[str, int | str]:
    records = discover_records(data_root, split)
    if max_records:
        # Uniformly span the sorted list rather than taking one scene-prefix block.
        indices = np.linspace(0, len(records) - 1, min(max_records, len(records)), dtype=int)
        records = [records[index] for index in indices]
    destination = output_root / "patches" / split
    destination.mkdir(parents=True, exist_ok=True)
    completed = 0
    # Processes, not threads: LibRaw decoding holds the GIL often enough that a
    # thread pool saturates only a few cores.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(
            build_record, record_index, record, destination, split,
            patches_per_record, patch_size, seed,
        ) for record_index, record in enumerate(records)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(records):
                print(
                    f"PATCH_PROGRESS split={split} records={completed}/{len(records)} "
                    f"patches={len(records) * patches_per_record}",
                    flush=True,
                )
    return {
        "split": split,
        "records": len(records),
        "patches": len(records) * patches_per_record,
        "patch_size": patch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode SID RAW once into reusable 16-bit training patches")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--train-patches", type=int, default=2)
    parser.add_argument("--eval-patches", type=int, default=1)
    parser.add_argument("--max-train-records", type=int, default=800)
    parser.add_argument("--max-eval-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    reports = []
    for offset, split in enumerate(("train", "val", "test")):
        reports.append(build_split(
            Path(args.data), Path(args.output), split,
            args.train_patches if split == "train" else args.eval_patches,
            args.patch_size,
            args.max_train_records if split == "train" else args.max_eval_records,
            args.seed + offset,
            args.workers,
        ))
    manifest = Path(args.output) / "patches" / "manifest.json"
    manifest.write_text(json.dumps(reports, indent=2) + "\n")
    print("PATCH_DATASET_COMPLETE " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
