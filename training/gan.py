"""Conditional PatchGAN for direct SID students.

The discriminator scores overlapping patches of (noisy input, output) pairs,
so detail that the noisy input does not support is penalized rather than
rewarded. Least-squares GAN losses; spectral normalization for stability.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, width: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for index, out in enumerate((width, width * 2, width * 4)):
            layers += [spectral_norm(nn.Conv2d(channels, out, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
            channels = out
        layers += [spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)), nn.LeakyReLU(0.2, inplace=True),
                   spectral_norm(nn.Conv2d(channels, 1, 3, padding=1))]
        self.network = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((condition, image), dim=1))


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return (fake_scores - 1).square().mean()


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return ((real_scores - 1).square().mean() + fake_scores.square().mean()) / 2
