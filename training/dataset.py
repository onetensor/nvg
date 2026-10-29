from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
import torch
from PIL import Image
from torch.utils.data import Dataset

from .synthetic_isp import random_camera_pipeline


EXPOSURE = re.compile(r"_([0-9.]+)s\.", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    input_path: Path
    target_path: Path
    raw_path: Path | None
    gain: float


def exposure(path: Path) -> float:
    match = EXPOSURE.search(path.name)
    if not match:
        raise ValueError(f"No exposure time in filename: {path.name}")
    return float(match.group(1))


def pack_sony(raw: rawpy.RawPy) -> np.ndarray:
    image = raw.raw_image_visible.astype(np.float32)
    image = np.maximum(image - 512.0, 0.0) / (16383.0 - 512.0)
    height, width = image.shape
    return np.stack(
        (image[0:height:2, 0:width:2], image[0:height:2, 1:width:2],
         image[1:height:2, 1:width:2], image[1:height:2, 0:width:2]),
        axis=0,
    )


def raw_rgb(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16,
        ).astype(np.float32) / 65535.0


def image_rgb(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".arw", ".dng", ".nef", ".raf"}:
        return raw_rgb(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def discover_records(root: Path, split: str) -> list[Record]:
    jsonl_candidates = [root / f"{split}.jsonl", root / "manifests" / f"{split}.jsonl"]
    for manifest in jsonl_candidates:
        if manifest.exists():
            records = []
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                item: dict[str, Any] = json.loads(line)
                records.append(Record(
                    input_path=root / item["input"],
                    target_path=root / item["target"],
                    raw_path=root / item["raw"] if item.get("raw") else None,
                    gain=float(item.get("gain", 30)),
                ))
            return records

    list_candidates = list(root.rglob(f"Sony_{split}_list.txt"))
    if list_candidates:
        base = list_candidates[0].parent
        records = []
        for line in list_candidates[0].read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            short_relative = parts[0].removeprefix("./")
            long_relative = parts[1].removeprefix("./")
            short_path = base / short_relative
            long_path = base / long_relative
            if not short_path.exists():
                short_path = base / short_relative.removeprefix("Sony/")
            if not long_path.exists():
                long_path = base / long_relative.removeprefix("Sony/")
            ratio = min(exposure(long_path) / exposure(short_path), 300.0)
            records.append(Record(short_path, long_path, short_path, ratio))
        return records

    # Official archive fallback when list files are absent.
    short = next(iter(root.rglob("short")), None)
    long = next(iter(root.rglob("long")), None)
    if short and long:
        prefix = {"train": "0", "val": "2", "test": "1"}[split]
        long_by_id = {path.name[:5]: path for path in long.glob(f"{prefix}*.ARW")}
        records = []
        for short_path in sorted(short.glob(f"{prefix}*.ARW")):
            long_path = long_by_id.get(short_path.name[:5])
            if long_path:
                records.append(Record(short_path, long_path, short_path,
                                      min(exposure(long_path) / exposure(short_path), 300.0)))
        return records
    raise FileNotFoundError(f"No supported {split} manifest or Sony short/long tree under {root}")


class AdapterDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: str | Path, split: str, patch_size: int = 512, augment: bool = True) -> None:
        self.root = Path(root)
        self.patches = sorted((self.root / "patches" / split).glob("*.npz"))
        self.records = [] if self.patches else discover_records(self.root, split)
        self.patch_size = patch_size
        self.augment = augment
        if not self.records and not self.patches:
            raise ValueError(f"No {split} records found in {self.root}")

    def __len__(self) -> int:
        return len(self.patches) if self.patches else len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.patches:
            return self._cached_patch(index)
        record = self.records[index]
        input_rgb = image_rgb(record.input_path)
        target_rgb = image_rgb(record.target_path)
        raw_value: np.ndarray | None = None
        if record.raw_path:
            with rawpy.imread(str(record.raw_path)) as raw:
                raw_value = pack_sony(raw)

        height = min(input_rgb.shape[0], target_rgb.shape[0])
        width = min(input_rgb.shape[1], target_rgb.shape[1])
        size = min(self.patch_size, height - height % 32, width - width % 32)
        if size < 32:
            raise ValueError(f"Image is too small for SID: {record.input_path}")
        if self.augment:
            y = random.randrange(0, height - size + 1)
            x = random.randrange(0, width - size + 1)
        else:
            y = max(0, (height - size) // 2)
            x = max(0, (width - size) // 2)
        y -= y % 2
        x -= x % 2
        input_crop = torch.from_numpy(input_rgb[y:y + size, x:x + size]).permute(2, 0, 1)
        target_crop = torch.from_numpy(target_rgb[y:y + size, x:x + size]).permute(2, 0, 1)
        if raw_value is not None:
            raw_crop = torch.from_numpy(raw_value[:, y // 2:(y + size) // 2, x // 2:(x + size) // 2])
        else:
            raw_crop = torch.zeros(4, size // 2, size // 2)

        if self.augment:
            input_crop = random_camera_pipeline(input_crop)
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [torch.flip(value, (-1,)) for value in (input_crop, target_crop, raw_crop)]
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [torch.flip(value, (-2,)) for value in (input_crop, target_crop, raw_crop)]
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [value.transpose(-1, -2) for value in (input_crop, target_crop, raw_crop)]
        else:
            from .synthetic_isp import inverse_srgb
            input_crop = inverse_srgb(input_crop.clamp(0, 1))
        return {
            "input": input_crop.contiguous(),
            "target": target_crop.contiguous(),
            "raw": raw_crop.contiguous(),
            "gain": torch.tensor(record.gain, dtype=torch.float32),
            "has_raw": torch.tensor(record.raw_path is not None),
        }

    def _cached_patch(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.patches[index]) as data:
            input_crop = torch.from_numpy(data["input"].astype(np.float32) / 65535.0).permute(2, 0, 1)
            target_crop = torch.from_numpy(data["target"].astype(np.float32) / 65535.0).permute(2, 0, 1)
            raw_crop = torch.from_numpy(data["raw"].astype(np.float32) / 65535.0)
            gain = float(data["gain"])
            has_raw = bool(data["has_raw"])
        if self.augment:
            input_crop = random_camera_pipeline(input_crop)
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [torch.flip(value, (-1,)) for value in (input_crop, target_crop, raw_crop)]
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [torch.flip(value, (-2,)) for value in (input_crop, target_crop, raw_crop)]
            if random.random() < 0.5:
                input_crop, target_crop, raw_crop = [value.transpose(-1, -2) for value in (input_crop, target_crop, raw_crop)]
        else:
            from .synthetic_isp import inverse_srgb
            input_crop = inverse_srgb(input_crop.clamp(0, 1))
        return {
            "input": input_crop.contiguous(),
            "target": target_crop.contiguous(),
            "raw": raw_crop.contiguous(),
            "gain": torch.tensor(gain, dtype=torch.float32),
            "has_raw": torch.tensor(has_raw),
        }
