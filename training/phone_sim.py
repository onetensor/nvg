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
    """Separable Gaussian blur of every plane of an [..., H, W] tensor."""
    radius = max(1, int(round(sigma * 2.5)))
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    shape = image.shape
    value = image.reshape(-1, 1, shape[-2], shape[-1])
    value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    value = F.conv2d(F.pad(value, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return value.reshape(shape)


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
    """8x8 DCT quantization of every [..., H, W] plane."""
    *lead, height, width = plane.shape
    blocks = plane.reshape(*lead, height // 8, 8, width // 8, 8).transpose(-3, -2)
    coefficients = dct @ blocks @ dct.T
    coefficients = torch.round(coefficients / table) * table
    restored = dct.T @ coefficients @ dct
    return restored.transpose(-3, -2).reshape(*lead, height, width)


def jpeg_like(encoded: torch.Tensor, quality: int) -> torch.Tensor:
    """Baseline-JPEG-style compression (YCbCr, 4:2:0, 8x8 DCT quantization) in torch.

    Stands in for H.264/HEVC block artifacts and chroma loss. ``encoded`` is
    [..., 3, H, W] with H and W multiples of 16.
    """
    device, dtype = encoded.device, encoded.dtype
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    luma_table = torch.tensor(_LUMA_TABLE, dtype=dtype, device=device).view(8, 8)
    chroma_table = torch.tensor(_CHROMA_TABLE, dtype=dtype, device=device).view(8, 8)
    luma_table = torch.clamp(torch.floor((luma_table * scale + 50) / 100), 1, 255)
    chroma_table = torch.clamp(torch.floor((chroma_table * scale + 50) / 100), 1, 255)
    dct = _dct_matrix(device, dtype)

    lead = encoded.shape[:-3]
    height, width = encoded.shape[-2:]
    red, green, blue = (encoded * 255).unbind(-3)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue

    def resample(plane: torch.Tensor, down: bool) -> torch.Tensor:
        flat = plane.reshape(-1, 1, *plane.shape[-2:])
        flat = F.avg_pool2d(flat, 2) if down else F.interpolate(flat, scale_factor=2, mode="bilinear", align_corners=False)
        return flat.reshape(*lead, *flat.shape[-2:])

    y = _quantize_blocks(y - 128, luma_table, dct) + 128
    cb = resample(_quantize_blocks(resample(cb, True), chroma_table, dct), False)
    cr = resample(_quantize_blocks(resample(cr, True), chroma_table, dct), False)
    rgb = torch.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), dim=-3)
    assert rgb.shape[-2:] == (height, width)
    return (torch.round(rgb.clamp(0, 255)) / 255).to(dtype)


def _luma_chroma(encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, green, blue = encoded.unbind(-3)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    return y, blue - y, red - y


def _from_luma_chroma(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    red = y + cr
    blue = y + cb
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    return torch.stack((red, green, blue), dim=-3)


def _draw_isp(rng: random.Random, device: torch.device) -> dict[str, object]:
    """One burst's phone ISP settings (shared by every frame and the clean reference)."""
    return {
        "white_balance": torch.tensor(
            [rng.uniform(0.85, 1.18), rng.uniform(0.94, 1.06), rng.uniform(0.82, 1.2)], device=device),
        "matrix": torch.eye(3, device=device) + torch.tensor(
            [[rng.gauss(0, 0.02) for _ in range(3)] for _ in range(3)], device=device),
        "gamma": rng.uniform(0.85, 1.15),
        # Local tone mapping lifts shadows; a global log curve stands in.
        "lift": rng.uniform(1.0, 8.0) if rng.random() < 0.5 else None,
        "black": rng.uniform(0.0, 0.03),
        "white": rng.uniform(0.9, 1.05),
        "chroma_blur": rng.uniform(0.8, 2.5) if rng.random() < 0.5 else None,
        "luma_blur": rng.uniform(0.4, 1.2) if rng.random() < 0.45 else None,
        "sharpen": (rng.uniform(0.3, 1.2), rng.uniform(0.6, 1.5)) if rng.random() < 0.5 else None,
        "jpeg": rng.randint(55, 92) if rng.random() < 0.6 else None,
    }


def _phone_tone(value: torch.Tensor, isp: dict[str, object] | None) -> torch.Tensor:
    """The signal-independent part of the ISP: color, encoding and tone curve."""
    if isp is None:
        return srgb_encode(value.clamp(0, 1))
    value = value * isp["white_balance"][:, None, None]
    value = torch.einsum("ij,...jhw->...ihw", isp["matrix"], value)
    encoded = srgb_encode(value.clamp(0, 1)).pow(isp["gamma"])
    if isp["lift"] is not None:
        encoded = torch.log1p(isp["lift"] * encoded) / math.log1p(isp["lift"])
    return ((encoded - isp["black"]) / (isp["white"] - isp["black"])).clamp(0, 1)


def simulate_phone_burst(
    shorts_encoded: torch.Tensor,
    ratios: torch.Tensor,
    long_encoded: torch.Tensor,
    rng: random.Random,
    downsample: int,
    extra_noise: float | None = None,
    isp: bool = True,
    repeated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulates iPhone video frames of a static scene.

    ``shorts_encoded`` is [K, 3, H, W]: independent short exposures of one
    scene. One set of phone exposure and ISP parameters applies to the whole
    burst, as it would to consecutive video frames; noise and compression are
    per frame. ``extra_noise`` is the median photon count of an added
    shot/read-noise stage (None draws it at random during training, 0 disables
    it). ``isp=False`` keeps only the sRGB encoding. ``repeated`` marks bursts
    that reuse frames; they always get the added noise so repeats stay
    independent.

    Returns (runtime linear RGB frames [K, 3, h, w], the SID long exposure as
    BT.709 [3, h, w], clean runtime linear RGB [3, h, w]). The last is the long
    exposure through the same phone exposure and color/tone ISP as the current
    (last) frame but without noise, noise reduction, sharpening, quantization
    or compression: what the model should see if the frame were noise-free.
    """
    device = shorts_encoded.device
    linear = libraw_decode(shorts_encoded)

    # iOS auto-exposure brightens the dark frames but leaves 2-100x for our
    # gain; each short is normalized by its own exposure ratio.
    exposure = rng.uniform(0.3, 2.0)
    phone_gain = (ratios.to(device) / 10 ** exposure).clamp_min(1.0)
    value = linear * phone_gain[:, None, None, None]

    # A phone sensor at 1/30 s collects far fewer photons than the Sony at
    # 0.033-0.1 s. Add shot (Poisson) and read (Gaussian) noise calibrated by
    # the photon count at the median signal to cover darker captures.
    photons = extra_noise
    if photons is None:
        photons = 10 ** rng.uniform(math.log10(1.5), math.log10(80)) if repeated or rng.random() < 0.6 else 0.0
    elif repeated and not photons:
        photons = 40.0
    if photons:
        scale = photons / value.median().clamp_min(1e-4)
        read = 10 ** rng.uniform(math.log10(0.3), math.log10(4.0)) if extra_noise is None else 1.0
        value = (torch.poisson((value * scale).clamp_min(0)) + torch.randn_like(value) * read) / scale

    # Camera ISP: color and tone, then noise reduction (chroma, luma) and
    # sharpening, which act on the noisy frames only.
    settings = _draw_isp(rng, device) if isp else None
    encoded = _phone_tone(value, settings)
    if settings is not None:
        y, cb, cr = _luma_chroma(encoded)
        if settings["chroma_blur"] is not None:
            cb, cr = _gaussian_blur(cb, settings["chroma_blur"]), _gaussian_blur(cr, settings["chroma_blur"])
        if settings["luma_blur"] is not None:
            y = _gaussian_blur(y, settings["luma_blur"])
        if settings["sharpen"] is not None:
            amount, sigma = settings["sharpen"]
            y = y + amount * (y - _gaussian_blur(y, sigma))
        encoded = _from_luma_chroma(y, cb, cr).clamp(0, 1)

    # 8-bit video, often with block compression.
    encoded = torch.round(encoded * 255) / 255
    if settings is not None and settings["jpeg"] is not None:
        encoded = jpeg_like(encoded, settings["jpeg"])

    # Runtime: inverse sRGB, then area downsampling in linear light.
    runtime = inverse_srgb(encoded)
    long_linear = libraw_decode(long_encoded)
    current_gain = float(phone_gain[-1] / ratios[-1])
    clean = inverse_srgb(_phone_tone(long_linear * current_gain, settings))
    if downsample > 1:
        runtime = F.avg_pool2d(runtime, downsample)
        long_linear = F.avg_pool2d(long_linear[None], downsample)[0]
        clean = F.avg_pool2d(clean[None], downsample)[0]
    return runtime.contiguous(), libraw_encode(long_linear).contiguous(), clean.contiguous()


def simulate_phone_capture(
    short_encoded: torch.Tensor,
    long_encoded: torch.Tensor,
    exposure_ratio: float,
    rng: random.Random,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-frame capture with the original (pre-burst) degradation: no extra noise."""
    frames, target, _ = simulate_phone_burst(
        short_encoded[None], torch.tensor([exposure_ratio]), long_encoded, rng, downsample, extra_noise=0.0)
    return frames[0], target


def emulate_accumulation(
    frames: torch.Tensor,
    max_frames: int,
    rng: random.Random,
    motion: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors src/temporal.ts on [K, 3, h, w] linear frames (oldest first).

    The runtime averages each packed 2x2 cell over up to ``max_frames`` frames
    and restarts a cell where it detects motion. ``motion`` restarts random
    regions at random times so the model sees spatially varying noise.
    Returns (accumulated [3, h, w], per-cell frame count [h/2, w/2]).
    """
    count_shape = (frames.shape[-2] // 2, frames.shape[-1] // 2)
    history = frames[0].clone()
    count = torch.ones(count_shape, device=frames.device)
    resets = []
    if motion and frames.shape[0] > 1:
        for _ in range(rng.randint(1, 3)):
            mask = torch.zeros(count_shape, dtype=torch.bool, device=frames.device)
            h = rng.randint(count_shape[0] // 8, count_shape[0] // 2)
            w = rng.randint(count_shape[1] // 8, count_shape[1] // 2)
            y = rng.randrange(0, count_shape[0] - h + 1)
            x = rng.randrange(0, count_shape[1] - w + 1)
            mask[y:y + h, x:x + w] = True
            resets.append((rng.randrange(1, frames.shape[0]), mask))
    for step in range(1, frames.shape[0]):
        count = (count + 1).clamp_max(max_frames)
        for time, mask in resets:
            if time == step:
                count = torch.where(mask, torch.ones_like(count), count)
        weight = F.interpolate(count[None, None], scale_factor=2, mode="nearest")[0]
        history = history + (frames[step] - history) / weight
    return history, count


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


def _uint16_bits(array: np.ndarray) -> torch.Tensor:
    """uint16 HWC -> CHW tensor of the same bits; torch lacks full uint16 support."""
    return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).permute(2, 0, 1)


def _from_uint16_bits(value: torch.Tensor) -> torch.Tensor:
    return (value.to(torch.int32) & 0xFFFF).to(torch.float32) / 65535.0


ACCUMULATION_CAPS = (1, 2, 4, 8, 16, 32)
# Sequences longer than a cached burst (at most 8 real frames) reuse frames;
# each reuse gets fresh simulated sensor noise (see ``simulate_phone_burst``),
# which keeps the reported frame count truthful for a "hold still" mode.
SEQUENCE_LENGTHS = (1, 2, 3, 4, 6, 8, 8, 16, 32)


def _sequence(rng: random.Random, available: int, length: int, shuffle: bool) -> list[int]:
    """Frame indices for a sequence: each real frame once, then repeats."""
    order = list(range(available))
    if shuffle:
        rng.shuffle(order)
    sequence = order[:length]
    while len(sequence) < length:
        sequence.append(rng.randrange(available) if shuffle else len(sequence) % available)
    return sequence


class BurstDataset(Dataset[dict[str, torch.Tensor]]):
    """Aligned SID bursts (see build_burst_dataset). Collate with ``collate_list``.

    Training draws the runtime downsampling, crop, frame subset and order;
    evaluation is deterministic: ``eval_frames`` frames at ``eval_downsample``.
    """

    def __init__(self, root: str | Path, split: str, output_size: int = 256, augment: bool = True,
                 eval_downsample: int = 2, eval_frames: int = 1, seed: int = 0) -> None:
        self.files = sorted((Path(root) / "bursts" / split).glob("*.npz"))
        self.files = [path for path in self.files if not path.name.endswith(".partial.npz")]
        # Teacher outline maps from build_edge_targets, when present.
        self.outlines = Path(root) / "edges" / split
        if not self.outlines.is_dir():
            self.outlines = None
        if not self.files:
            raise ValueError(f"No cached {split} bursts under {root}")
        self.output_size = output_size
        self.augment = augment
        self.eval_downsample = eval_downsample
        self.eval_frames = eval_frames
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + index)
        with np.load(self.files[index]) as data:
            available = sum(1 for key in data.files if key.startswith("short_"))
            ratios_all = data["ratios"]
            length = rng.choice(SEQUENCE_LENGTHS) if self.augment else self.eval_frames
            sequence = _sequence(rng, available, length, shuffle=self.augment)
            # Only distinct frames are decoded and transferred; ``order`` expands them.
            chosen = sorted(set(sequence))
            order = torch.tensor([chosen.index(frame) for frame in sequence])
            downsample = rng.choice((1, 2)) if self.augment else self.eval_downsample
            crop = self.output_size * downsample
            size = data["long"].shape[0]
            if crop > size:
                raise ValueError(f"{self.output_size}px output at {downsample}x needs crops beyond {size}px")
            if self.augment:
                y = rng.randrange(0, size - crop + 1) // 2 * 2
                x = rng.randrange(0, size - crop + 1) // 2 * 2
            else:
                y = x = (size - crop) // 2 // 2 * 2
            window = (slice(y, y + crop), slice(x, x + crop))
            long = _uint16_bits(data["long"][window])
            outline = None
            if self.outlines is not None:
                outline = torch.from_numpy(np.load(self.outlines / self.files[index].with_suffix(".npy").name)[window].copy())[None]
            shorts = torch.stack([_uint16_bits(data[f"short_{frame:02d}"][window]) for frame in chosen])
            ratios = torch.from_numpy(ratios_all[chosen].astype(np.float32))
        if self.augment:
            transforms = [rng.random() < 0.5 for _ in range(3)]
            for flip, dims in zip(transforms[:2], ((-1,), (-2,))):
                if flip:
                    long, shorts = long.flip(dims), shorts.flip(dims)
                    outline = outline.flip(dims) if outline is not None else None
            if transforms[2]:
                long, shorts = long.transpose(-1, -2), shorts.transpose(-1, -2)
                outline = outline.transpose(-1, -2) if outline is not None else None
        extra = {"outline": outline.contiguous()} if outline is not None else {}
        return {
            **extra,
            "long": long.contiguous(),
            "shorts": shorts.contiguous(),
            "ratios": ratios,
            "order": order,
            "downsample": torch.tensor(downsample),
            "seed": torch.tensor(rng.getrandbits(62) if self.augment else self.seed * 1_000_003 + index),
        }


def collate_list(samples: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Bursts differ in frame count and crop size, so batches stay lists."""
    return samples


def degrade_bursts(
    samples: list[dict[str, torch.Tensor]],
    device: torch.device,
    augment: bool,
    extra_noise: float | None = None,
    accumulation_cap: int | None = None,
    target: str = "phone",
    outlines: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulated captures accumulated like the runtime.

    Returns (accumulated linear RGB [B, 3, h, w], targets [B, 3, h, w], gains
    [B], per-cell frame counts [B, h/2, w/2]). Evaluation uses a plain average
    of all frames (``accumulation_cap`` defaults to the frame count) and fixed
    ``extra_noise``.

    ``target="phone"`` is the noise-free input as the model receives it (clean
    phone rendering times the runtime gain, sRGB-encoded): the model restores
    the phone's own image instead of guessing an unidentifiable ISP.
    ``target="sid"`` is the SID long exposure as LibRaw renders it.

    Pass a list as ``outlines`` to collect each sample's teacher outline map,
    area-downsampled like the image, as [1, h, w] tensors in [0, 1].
    """
    if target not in ("phone", "sid"):
        raise ValueError(f"Unknown target {target!r}")
    inputs, references, counts, currents, jitter = [], [], [], [], []
    for sample in samples:
        rng = random.Random(int(sample["seed"]))
        order = sample["order"]
        shorts = _from_uint16_bits(sample["shorts"].to(device, non_blocking=True))[order.to(device)]
        long = _from_uint16_bits(sample["long"].to(device, non_blocking=True))
        frames, sid_target, clean = simulate_phone_burst(
            shorts, sample["ratios"][order], long, rng, int(sample["downsample"]),
            extra_noise=None if augment else (extra_noise or 0.0),
            repeated=len(set(order.tolist())) < len(order),
        )
        cap = rng.choice(ACCUMULATION_CAPS) if augment else (accumulation_cap or frames.shape[0])
        accumulated, count = emulate_accumulation(frames, cap, rng, motion=augment and rng.random() < 0.5)
        inputs.append(accumulated)
        references.append(clean if target == "phone" else sid_target)
        if outlines is not None:
            outline = sample["outline"].to(device, non_blocking=True).float()[None] / 255.0
            downsample = int(sample["downsample"])
            outlines.append(F.avg_pool2d(outline, downsample)[0] if downsample > 1 else outline[0])
        counts.append(count)
        # The runtime measures gain on the current frame, before accumulation.
        currents.append(frames[-1])
        jitter.append(2 ** rng.uniform(-0.35, 0.35) if augment else 1.0)
    gains = (runtime_auto_gain(torch.stack(currents)) * torch.tensor(jitter, device=device)).clamp(1.0, MAX_GAIN)
    targets = torch.stack(references)
    if target == "phone":
        targets = srgb_encode((targets * gains[:, None, None, None]).clamp(0, 1))
    return torch.stack(inputs), targets, gains, torch.stack(counts)
