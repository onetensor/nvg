"""A dedicated object-outline network that runs on top of a frozen direct SID.

The 1x1 outline head on a slim model's features produced thick, soft outlines
and lost to plain Sobel edges of the denoised output. This network instead sees
crisp evidence at the display resolution: the denoised luminance, its Sobel
magnitude, the noisy gained luminance and the per-pixel noise level. It works
at half resolution (space-to-depth in, PixelShuffle out, like SID itself) with
a small U-Net for object context, so it can learn which edges are object
outlines and which are texture or leftover noise.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from training.sid_model import LUMA_WEIGHTS, SIDUNet

OUTLINE_INPUTS = 4


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of [B, 1, H, W] images (replicate padding).

    Written without channel reductions: ReduceSum over channels is very slow in
    ONNX Runtime's WebGPU kernels, so gx^2 + gy^2 uses the two planes directly.
    """
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype)
    kernels = torch.stack((kernel_x, kernel_x.T))[:, None]
    gradients = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernels)
    return (gradients[:, :1].square() + gradients[:, 1:].square() + 1e-8).sqrt()


def outline_inputs(sid_input: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """[denoised luma, its Sobel, noisy gained luma, noise level] at display resolution.

    ``sid_input`` is the direct model's 13-channel input (packed, gained RGB
    plus the 1/sqrt(frames) plane) and ``image`` its [B, 1, H, W] output.
    """
    noisy_rgb = F.pixel_shuffle(sid_input[:, :12], 2)
    # A fixed 1x1 convolution rather than a channel sum (see sobel_magnitude).
    weights = torch.tensor(LUMA_WEIGHTS, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    noisy = F.conv2d(noisy_rgb, weights).clamp(0, 1).pow(1 / 2.2)
    noise_level = F.interpolate(sid_input[:, 12:13], scale_factor=2, mode="nearest")
    clean = image.clamp(0, 1)
    return torch.cat((clean, sobel_magnitude(clean) / 2, noisy, noise_level), dim=1)


class OutlineNet(nn.Module):
    def __init__(self, widths: tuple[int, int, int] = (16, 24, 32)) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.widths = widths
        activate = nn.LeakyReLU(0.2, inplace=True)
        # Space-to-depth keeps the full-resolution evidence at half resolution.
        self.enter = nn.Sequential(nn.Conv2d(OUTLINE_INPUTS * 4, w1, 3, padding=1), activate,
                                   nn.Conv2d(w1, w1, 3, padding=1), activate)
        self.down1 = nn.Sequential(nn.Conv2d(w1, w2, 3, stride=2, padding=1), activate,
                                   nn.Conv2d(w2, w2, 3, padding=1), activate)
        self.down2 = nn.Sequential(nn.Conv2d(w2, w3, 3, stride=2, padding=1), activate,
                                   nn.Conv2d(w3, w3, 3, padding=1), activate)
        self.up2 = nn.Sequential(nn.Conv2d(w3 + w2, w2, 3, padding=1), activate)
        self.up1 = nn.Sequential(nn.Conv2d(w2 + w1, w1, 3, padding=1), activate)
        # Four sub-pixel logits per half-resolution cell.
        self.exit = nn.Conv2d(w1 + OUTLINE_INPUTS * 4, 4, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        packed = F.pixel_unshuffle(inputs, 2)
        level1 = self.enter(packed)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        # Fixed x2 scales (inputs are multiples of 8) keep shape arithmetic out
        # of the exported graph.
        up = F.interpolate(level3, scale_factor=2.0, mode="bilinear", align_corners=False)
        level2 = self.up2(torch.cat((up, level2), dim=1))
        up = F.interpolate(level2, scale_factor=2.0, mode="bilinear", align_corners=False)
        level1 = self.up1(torch.cat((up, level1), dim=1))
        return F.pixel_shuffle(self.exit(torch.cat((level1, packed), dim=1)), 2)


class OutlinedSID(nn.Module):
    """Direct SID plus outline network as one graph: [luma, outline logits]."""

    def __init__(self, sid: SIDUNet, outline: OutlineNet) -> None:
        super().__init__()
        if sid.in_channels != 13 or sid.out_channels != 4:
            raise ValueError("Expected a 13-channel mono direct model")
        self.sid = sid
        self.outline = outline

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        image = self.sid(inputs)
        return torch.cat((image, self.outline(outline_inputs(inputs, image))), dim=1)


def load_outline(path: str, device: torch.device | str = "cpu") -> OutlineNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = OutlineNet(tuple(payload["widths"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def tolerance_counts(probability: torch.Tensor, thin_target: torch.Tensor, thresholds: torch.Tensor,
                     radius: int = 2) -> torch.Tensor:
    """Boundary-benchmark style counts with a pixel tolerance, per threshold.

    Returns [4, T]: predicted pixels within ``radius`` of a reference outline,
    predicted pixels, reference pixels within ``radius`` of a prediction, and
    reference pixels. ``thin_target`` is binary one-pixel outlines.
    """
    size = 2 * radius + 1
    reference = thin_target > 0.5
    near_reference = F.max_pool2d(reference.float(), size, stride=1, padding=radius) > 0
    counts = torch.zeros(4, len(thresholds), device=probability.device)
    for index, threshold in enumerate(thresholds):
        predicted = probability > threshold
        near_predicted = F.max_pool2d(predicted.float(), size, stride=1, padding=radius) > 0
        counts[0, index] = (predicted & near_reference).sum()
        counts[1, index] = predicted.sum()
        counts[2, index] = (reference & near_predicted).sum()
        counts[3, index] = reference.sum()
    return counts


def best_f_score(counts: torch.Tensor, thresholds: torch.Tensor) -> dict[str, float]:
    precision = counts[0] / counts[1].clamp_min(1)
    recall = counts[2] / counts[3].clamp_min(1)
    f_score = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    best = int(f_score.argmax())
    return {"f": float(f_score[best]), "precision": float(precision[best]), "recall": float(recall[best]),
            "threshold": float(thresholds[best])}
