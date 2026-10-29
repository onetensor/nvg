"""Simulated iPhone video capture for training and evaluating direct SID models.

The SID pairs are a short (dark) and a long (reference) exposure of the same
scene, both developed by LibRaw to BT.709-gamma RGB. The runtime never sees a
100-300x underexposed frame with a known exposure ratio: iOS auto-exposure has
already brightened the video, its ISP has tone-mapped, denoised and compressed
it to 8 bits, and the web app then linearizes, area-downsamples and applies
its own histogram-based auto gain. This module reproduces that chain so the
model is trained on the problem it actually faces.

The degradation runs on whatever device its tensors are on. Training applies
it on the GPU per batch (``degrade_batch``) because the RunPod container's CPU
quota cannot keep the GPU fed; data loader workers only read and flip patches.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .synthetic_isp import inverse_srgb

MAX_GAIN = 300.0
PATCH_SIZE = 512


def libraw_decode(value: torch.Tensor) -> torch.Tensor:
    """Inverse of LibRaw's default BT.709 output gamma (power 0.45, toe slope 4.5)."""
    value = value.clamp(0, 1)
    return torch.where(value < 0.081, value / 4.5, ((value + 0.099) / 1.099).pow(1 / 0.45))


def libraw_encode(linear: torch.Tensor) -> torch.Tensor:
    linear = linear.clamp(0, 1)
    return torch.where(linear < 0.018, linear * 4.5, 1.099 * linear.clamp_min(1e-12).pow(0.45) - 0.099)


def srgb_encode(linear: torch.Tensor) -> torch.Tensor:
    linear = linear.clamp(0, 1)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(1e-12).pow(1 / 2.4) - 0.055)


def runtime_auto_gain(linear_rgb: torch.Tensor) -> torch.Tensor:
    """The steady-state target of src/autoGain.ts for each frame of a BCHW linear batch."""
    luminance = (0.2126 * linear_rgb[:, 0] + 0.7152 * linear_rgb[:, 1] + 0.0722 * linear_rgb[:, 2]).flatten(1)
    stride = luminance.shape[1] // 65_536 + 1
    percentiles = torch.quantile(luminance[:, ::stride], torch.tensor([0.5, 0.9], device=luminance.device), dim=1)
    p50, p90 = percentiles[0], percentiles[1]
    black = (luminance < 4 / 255).float().mean(dim=1)
    target = torch.minimum(0.18 / p50.clamp_min(0.0006), 0.82 / p90.clamp_min(0.001) * 1.35)
    target = torch.where(black > 0.55, target * 1.15, target)
    return target.clamp(1.0, MAX_GAIN)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(sigma * 2.5)))
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[0]
    value = image[None]
    value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode="replicate"),
                     kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    value = F.conv2d(F.pad(value, (0, 0, radius, radius), mode="replicate"),
                     kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)
    return value[0]


# Standard JPEG (ITU-T T.81 Annex K) quantization tables.
_LUMA_TABLE = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_CHROMA_TABLE = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    *([99] * 32),
]


def _dct_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(8, dtype=dtype, device=device)
    matrix = torch.cos((2 * index[None, :] + 1) * index[:, None] * math.pi / 16) * math.sqrt(2 / 8)
    matrix[0] = matrix[0] / math.sqrt(2)
    return matrix


def _quantize_blocks(plane: torch.Tensor, table: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
    height, width = plane.shape
    blocks = plane.reshape(height // 8, 8, width // 8, 8).permute(0, 2, 1, 3)
    coefficients = dct @ blocks @ dct.T
    coefficients = torch.round(coefficients / table) * table
    restored = dct.T @ coefficients @ dct
    return restored.permute(0, 2, 1, 3).reshape(height, width)


def jpeg_like(encoded: torch.Tensor, quality: int) -> torch.Tensor:
    """Baseline-JPEG-style compression (YCbCr, 4:2:0, 8x8 DCT quantization) in torch.

    Stands in for H.264/HEVC block artifacts and chroma loss. Height and width
    must be multiples of 16.
    """
    device, dtype = encoded.device, encoded.dtype
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    luma_table = torch.tensor(_LUMA_TABLE, dtype=dtype, device=device).view(8, 8)
    chroma_table = torch.tensor(_CHROMA_TABLE, dtype=dtype, device=device).view(8, 8)
    luma_table = torch.clamp(torch.floor((luma_table * scale + 50) / 100), 1, 255)
    chroma_table = torch.clamp(torch.floor((chroma_table * scale + 50) / 100), 1, 255)
    dct = _dct_matrix(device, dtype)

    red, green, blue = encoded * 255
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue
    cb = F.avg_pool2d(cb[None, None], 2)[0, 0]
    cr = F.avg_pool2d(cr[None, None], 2)[0, 0]
    y = _quantize_blocks(y - 128, luma_table, dct) + 128
    cb = _quantize_blocks(cb, chroma_table, dct)
    cr = _quantize_blocks(cr, chroma_table, dct)
    cb = F.interpolate(cb[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    cr = F.interpolate(cr[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    rgb = torch.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb))
    return (torch.round(rgb.clamp(0, 255)) / 255).to(dtype)


def simulate_phone_capture(
    short_encoded: torch.Tensor,
    long_encoded: torch.Tensor,
    exposure_ratio: float,
    rng: random.Random,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (runtime linear RGB input, BT.709-encoded target)."""
    device = short_encoded.device
    linear = libraw_decode(short_encoded)

    # iOS auto-exposure brightens the dark frame but leaves 2-100x for our gain.
    phone_gain = exposure_ratio / 10 ** rng.uniform(0.3, 2.0)
    value = linear * max(phone_gain, 1.0)

    # Camera ISP: white balance and color matrix error, tone curve, black
    # crush, highlight roll, optional noise-reduction blur.
    white_balance = torch.tensor(
        [rng.uniform(0.85, 1.18), rng.uniform(0.94, 1.06), rng.uniform(0.82, 1.2)], device=device)[:, None, None]
    value = value * white_balance
    matrix = torch.eye(3, device=device) + torch.tensor(
        [[rng.gauss(0, 0.02) for _ in range(3)] for _ in range(3)], device=device)
    value = torch.einsum("ij,jhw->ihw", matrix, value).clamp(0, 1)
    encoded = srgb_encode(value).pow(rng.uniform(0.85, 1.15))
    black = rng.uniform(0.0, 0.03)
    white = rng.uniform(0.9, 1.05)
    encoded = ((encoded - black) / (white - black)).clamp(0, 1)
    if rng.random() < 0.45:
        encoded = _gaussian_blur(encoded, rng.uniform(0.4, 1.2))

    # 8-bit video, often with block compression.
    encoded = torch.round(encoded * 255) / 255
    if rng.random() < 0.6:
        encoded = jpeg_like(encoded, rng.randint(55, 92))

    # Runtime: inverse sRGB, then area downsampling in linear light.
    runtime = inverse_srgb(encoded)
    target_linear = libraw_decode(long_encoded)
    if downsample > 1:
        runtime = F.avg_pool2d(runtime[None], downsample)[0]
        target_linear = F.avg_pool2d(target_linear[None], downsample)[0]
    return runtime.contiguous(), libraw_encode(target_linear).contiguous()


class PhoneSimDataset(Dataset[dict[str, torch.Tensor]]):
    """Full cached SID patches plus a per-sample seed; ``degrade_batch`` does the rest."""

    def __init__(self, root: str | Path, split: str, augment: bool = True, seed: int = 0) -> None:
        self.patches = sorted((Path(root) / "patches" / split).glob("*.npz"))
        if not self.patches:
            raise ValueError(f"No cached {split} patches under {root}")
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.patches[index]) as data:
            short = torch.from_numpy(data["input"].astype(np.float32) / 65535.0).permute(2, 0, 1)
            long = torch.from_numpy(data["target"].astype(np.float32) / 65535.0).permute(2, 0, 1)
            ratio = float(data["gain"])
        if short.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"{self.patches[index]} is not a {PATCH_SIZE}px patch")
        if self.augment:
            seed = random.getrandbits(62)
            if random.random() < 0.5:
                short, long = short.flip(-1), long.flip(-1)
            if random.random() < 0.5:
                short, long = short.flip(-2), long.flip(-2)
            if random.random() < 0.5:
                short, long = short.transpose(-1, -2), long.transpose(-1, -2)
        else:
            seed = self.seed * 1_000_003 + index
        return {
            "short": short.contiguous(),
            "long": long.contiguous(),
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "seed": torch.tensor(seed, dtype=torch.int64),
        }


def degrade_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    output_size: int,
    augment: bool,
    eval_downsample: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies the phone simulation per sample on ``device``: (inputs, targets, gains)."""
    shorts = batch["short"].to(device, non_blocking=True)
    longs = batch["long"].to(device, non_blocking=True)
    inputs, targets, jitter = [], [], []
    for index in range(shorts.shape[0]):
        rng = random.Random(int(batch["seed"][index]))
        downsample = rng.choice((1, 2)) if augment else eval_downsample
        crop = output_size * downsample
        if crop > PATCH_SIZE:
            raise ValueError(f"{output_size}px output at {downsample}x needs crops beyond {PATCH_SIZE}px")
        if augment:
            y = rng.randrange(0, PATCH_SIZE - crop + 1) // 2 * 2
            x = rng.randrange(0, PATCH_SIZE - crop + 1) // 2 * 2
        else:
            y = x = (PATCH_SIZE - crop) // 2 // 2 * 2
        runtime, target = simulate_phone_capture(
            shorts[index, :, y:y + crop, x:x + crop],
            longs[index, :, y:y + crop, x:x + crop],
            float(batch["ratio"][index]),
            rng,
            downsample,
        )
        # Auto gain is smoothed over time at runtime, so it is rarely exact.
        jitter.append(2 ** rng.uniform(-0.35, 0.35) if augment else 1.0)
        inputs.append(runtime)
        targets.append(target)
    inputs_batch = torch.stack(inputs)
    gains = runtime_auto_gain(inputs_batch) * torch.tensor(jitter, device=device)
    return inputs_batch, torch.stack(targets), gains.clamp(1.0, MAX_GAIN)
