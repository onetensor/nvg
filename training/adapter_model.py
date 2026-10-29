from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class PseudoRawAdapter(nn.Module):
    """Small processed-linear-RGB to unamplified pseudo-RAW translator.

    The learned branch is a bounded residual over the deterministic Bayer
    mapping. Zero-initializing its final projection makes step zero exactly the
    baseline instead of producing ~0.5 everywhere and saturating after SID's
    exposure multiplication.
    """

    def __init__(self, width: int = 64, residual_scale: float = 0.10) -> None:
        super().__init__()
        self.width = width
        self.residual_scale = residual_scale
        self.unshuffle = nn.PixelUnshuffle(2)
        self.network = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(width),
            ResidualBlock(width),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(width),
            nn.Conv2d(width, 4, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = manual_pseudo_bayer(inputs)
        residual = self.residual_scale * torch.tanh(self.network(self.unshuffle(inputs)))
        return torch.clamp(base + residual, 0.0, 1.0)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def manual_pseudo_bayer(linear_rgb: torch.Tensor) -> torch.Tensor:
    unshuffled = nn.functional.pixel_unshuffle(linear_rgb, 2)
    # PixelUnshuffle channel order is channel-major then (00, 01, 10, 11).
    return unshuffled[:, [0, 5, 11, 6], :, :]
