from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image


def inverse_srgb(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def random_camera_pipeline(encoded_rgb: torch.Tensor) -> torch.Tensor:
    """Randomized ISP/video damage followed by the runtime's inverse-sRGB step."""
    value = encoded_rgb.clamp(0, 1)
    device = value.device
    dtype = value.dtype

    wb = torch.tensor(
        [random.uniform(0.82, 1.22), random.uniform(0.92, 1.08), random.uniform(0.78, 1.25)],
        device=device,
        dtype=dtype,
    )[:, None, None]
    value = value * wb
    matrix = torch.eye(3, device=device, dtype=dtype) + torch.randn(3, 3, device=device, dtype=dtype) * 0.025
    value = torch.einsum("ij,jhw->ihw", matrix, value)
    gamma = random.uniform(0.82, 1.24)
    value = value.clamp(0, 1).pow(gamma)
    black = random.uniform(0, 0.018)
    white = random.uniform(0.88, 1.08)
    value = ((value - black) / max(white - black, 1e-3)).clamp(0, 1)

    if random.random() < 0.65:
        levels = random.choice((255, 255, 1023))
        value = torch.round(value * levels) / levels
    if random.random() < 0.35:
        # JPEG round-trip is intentionally occasional and off-graph in the input pipeline.
        array = (value.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="JPEG", quality=random.randint(55, 94), subsampling=2)
        stream.seek(0)
        decoded = np.asarray(Image.open(stream).convert("RGB"), dtype=np.float32) / 255.0
        value = torch.from_numpy(decoded).permute(2, 0, 1).to(device=device, dtype=dtype)
    return inverse_srgb(value.clamp(0, 1))
