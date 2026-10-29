from __future__ import annotations

import torch
from torch.nn import functional as F


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), window, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target).clamp_min(1e-12)
    return -10 * torch.log10(mse)
