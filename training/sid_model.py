from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SIDUNet(nn.Module):
    """NCHW equivalent of the official Sony SID TensorFlow network."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 12, 1)

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
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = SIDUNet().to(device)
    model.load_state_dict(state, strict=True)
    return model
