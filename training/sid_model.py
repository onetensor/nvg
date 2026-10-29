from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SIDUNet(nn.Module):
    """NCHW equivalent of the official Sony SID TensorFlow network.

    The defaults (4 packed Bayer channels, base width 32, RGB output) are the
    official architecture. ``in_channels=12`` accepts 2x2-packed linear RGB
    directly, a smaller ``width`` scales every level uniformly, and
    ``out_channels=4`` produces a single (luminance) output plane.
    """

    def __init__(self, in_channels: int = 4, width: int = 32, out_channels: int = 12) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.out_channels = out_channels
        c1, c2, c3, c4, c5 = (width * factor for factor in (1, 2, 4, 8, 16))
        self.conv1_1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv1_2 = nn.Conv2d(c1, c1, 3, padding=1)
        self.conv2_1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2_2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.conv3_1 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv3_2 = nn.Conv2d(c3, c3, 3, padding=1)
        self.conv4_1 = nn.Conv2d(c3, c4, 3, padding=1)
        self.conv4_2 = nn.Conv2d(c4, c4, 3, padding=1)
        self.conv5_1 = nn.Conv2d(c4, c5, 3, padding=1)
        self.conv5_2 = nn.Conv2d(c5, c5, 3, padding=1)
        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(c5, c4, 3, padding=1)
        self.conv6_2 = nn.Conv2d(c4, c4, 3, padding=1)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(c4, c3, 3, padding=1)
        self.conv7_2 = nn.Conv2d(c3, c3, 3, padding=1)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(c3, c2, 3, padding=1)
        self.conv8_2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.conv9_2 = nn.Conv2d(c1, c1, 3, padding=1)
        self.conv10 = nn.Conv2d(c1, out_channels, 1)

    @staticmethod
    def activate(value: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(value, negative_slope=0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv1 = self.activate(self.conv1_1(inputs))
        conv1 = self.activate(self.conv1_2(conv1))
        conv2 = self.activate(self.conv2_1(F.max_pool2d(conv1, 2)))
        conv2 = self.activate(self.conv2_2(conv2))
        conv3 = self.activate(self.conv3_1(F.max_pool2d(conv2, 2)))
        conv3 = self.activate(self.conv3_2(conv3))
        conv4 = self.activate(self.conv4_1(F.max_pool2d(conv3, 2)))
        conv4 = self.activate(self.conv4_2(conv4))
        conv5 = self.activate(self.conv5_1(F.max_pool2d(conv4, 2)))
        conv5 = self.activate(self.conv5_2(conv5))
        conv6 = self.activate(self.conv6_1(torch.cat((self.up6(conv5), conv4), dim=1)))
        conv6 = self.activate(self.conv6_2(conv6))
        conv7 = self.activate(self.conv7_1(torch.cat((self.up7(conv6), conv3), dim=1)))
        conv7 = self.activate(self.conv7_2(conv7))
        conv8 = self.activate(self.conv8_1(torch.cat((self.up8(conv7), conv2), dim=1)))
        conv8 = self.activate(self.conv8_2(conv8))
        conv9 = self.activate(self.conv9_1(torch.cat((self.up9(conv8), conv1), dim=1)))
        conv9 = self.activate(self.conv9_2(conv9))
        return F.pixel_shuffle(self.conv10(conv9), 2)


def load_sid(path: str, device: torch.device | str = "cpu") -> SIDUNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    wrapped = isinstance(payload, dict) and "state_dict" in payload
    state = payload["state_dict"] if wrapped else payload
    model = SIDUNet(
        in_channels=int(payload.get("in_channels", 4)) if wrapped else 4,
        width=int(payload.get("width", 32)) if wrapped else 32,
        out_channels=int(payload.get("out_channels", 12)) if wrapped else 12,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model


# PixelUnshuffle(2) emits channel-major (00, 01, 10, 11) planes, so the
# deterministic RGGB baseline (R00, G01, B11, G10) is these packed channels.
PSEUDO_BAYER_CHANNELS = (0, 5, 11, 6)


def packed_linear_rgb(linear_rgb: torch.Tensor, gain: torch.Tensor | float) -> torch.Tensor:
    """Runtime input for direct models: 2x2-packed linear RGB, amplified and clipped."""
    return (F.pixel_unshuffle(linear_rgb, 2) * gain).clamp(0, 1)


def direct_input(
    model: SIDUNet,
    linear_rgb: torch.Tensor,
    gain: torch.Tensor | float,
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed input, plus a noise-level plane 1/sqrt(frames averaged) for 13-channel models."""
    packed = packed_linear_rgb(linear_rgb, gain)
    if model.in_channels == 12:
        return packed
    if model.in_channels != 13:
        raise ValueError(f"Unsupported direct model with {model.in_channels} input channels")
    if counts is None:
        counts = torch.ones(packed.shape[0], *packed.shape[-2:], device=packed.device)
    return torch.cat((packed, counts[:, None].clamp_min(1).rsqrt()), dim=1)


# Rec. 709 luma weights, applied to display-encoded RGB.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] display RGB -> [B, 1, H, W] luma."""
    weights = torch.tensor(LUMA_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights[None, :, None, None]).sum(dim=1, keepdim=True)


def to_mono(source: SIDUNet) -> SIDUNet:
    """A single-output copy whose output is exactly the luma of the source's (unclamped) RGB.

    PixelShuffle(2) maps head channel c * 4 + k to color c at sub-pixel k, and
    luma is linear, so the new head is the luma-weighted sum of the color heads.
    """
    if source.out_channels != 12:
        raise ValueError("Source must produce RGB")
    model = SIDUNet(in_channels=source.in_channels, width=source.width, out_channels=4)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    weight, bias = state["conv10.weight"], state["conv10.bias"]
    state["conv10.weight"] = sum(w * weight[c * 4:(c + 1) * 4] for c, w in enumerate(LUMA_WEIGHTS))
    state["conv10.bias"] = sum(w * bias[c * 4:(c + 1) * 4] for c, w in enumerate(LUMA_WEIGHTS))
    model.load_state_dict(state, strict=True)
    return model


def prune_width(source: SIDUNet, width: int) -> SIDUNet:
    """Structured pruning to a narrower U-Net that keeps the source's strongest channels.

    Every level keeps the output channels with the largest filter L1 norms, and
    each following layer keeps the matching input channels, including both
    halves of the decoder's skip concatenations. The result is a warm start for
    distillation, not a finished model.
    """
    if width >= source.width:
        raise ValueError("Pruning must reduce the width")
    state = source.state_dict()
    pruned: dict[str, torch.Tensor] = {}
    full = [source.width * factor for factor in (1, 2, 4, 8, 16)]
    kept = [width * factor for factor in (1, 2, 4, 8, 16)]

    def strongest(norms: torch.Tensor, count: int) -> torch.Tensor:
        return torch.topk(norms, count).indices.sort().values

    def conv(name: str, inputs: torch.Tensor, count: int | None) -> torch.Tensor:
        weight = state[f"{name}.weight"][:, inputs]
        outputs = (torch.arange(weight.shape[0]) if count is None
                   else strongest(weight.abs().sum(dim=(1, 2, 3)), count))
        pruned[f"{name}.weight"] = weight[outputs].clone()
        pruned[f"{name}.bias"] = state[f"{name}.bias"][outputs].clone()
        return outputs

    def upsample(name: str, inputs: torch.Tensor, count: int) -> torch.Tensor:
        weight = state[f"{name}.weight"][inputs]  # ConvTranspose: [in, out, k, k]
        outputs = strongest(weight.abs().sum(dim=(0, 2, 3)), count)
        pruned[f"{name}.weight"] = weight[:, outputs].clone()
        return outputs

    features = torch.arange(source.in_channels)
    skips = []
    for level in range(5):
        features = conv(f"conv{level + 1}_1", features, kept[level])
        features = conv(f"conv{level + 1}_2", features, kept[level])
        skips.append(features)
    for decoder, level in zip(range(6, 10), range(3, -1, -1)):
        upsampled = upsample(f"up{decoder}", features, kept[level])
        # Decoder input is cat(upsampled, skip); skip channels follow full[level].
        features = conv(f"conv{decoder}_1", torch.cat((upsampled, skips[level] + full[level])), kept[level])
        features = conv(f"conv{decoder}_2", features, kept[level])
    conv("conv10", features, None)

    model = SIDUNet(in_channels=source.in_channels, width=width, out_channels=source.out_channels)
    model.load_state_dict(pruned, strict=True)
    return model


def add_noise_level_input(source: SIDUNet) -> SIDUNet:
    """A 13-channel copy of a 12-channel direct model whose extra plane starts at zero weight."""
    if source.in_channels != 12:
        raise ValueError("Source must be a 12-channel direct model")
    model = SIDUNet(in_channels=13, width=source.width, out_channels=source.out_channels)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    weight = state["conv1_1.weight"]
    state["conv1_1.weight"] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
    model.load_state_dict(state, strict=True)
    return model


def direct_from_bayer(source: SIDUNet) -> SIDUNet:
    """A 12-channel SID that initially reproduces ``source`` on the deterministic pseudo-Bayer input."""
    if source.in_channels != 4:
        raise ValueError("Source SID must take 4 packed Bayer channels")
    model = SIDUNet(in_channels=12, width=source.width)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    weight = torch.zeros(source.width, 12, 3, 3, dtype=state["conv1_1.weight"].dtype)
    for bayer_channel, packed_channel in enumerate(PSEUDO_BAYER_CHANNELS):
        weight[:, packed_channel] = state["conv1_1.weight"][:, bayer_channel]
    state["conv1_1.weight"] = weight
    model.load_state_dict(state, strict=True)
    return model
