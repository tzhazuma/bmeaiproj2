from __future__ import annotations

import torch
from torch import Tensor, nn

from .unet import DoubleConv, DownBlock


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attention


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(out_channels, skip_channels, out_channels // 2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        skip = self.attn(x, skip)
        return self.conv(_concat_channels(skip, x))


def _concat_channels(left: Tensor, right: Tensor) -> Tensor:
    return torch.ops.aten.cat.default([left, right], 1)


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, features: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.stem = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.pool = nn.MaxPool2d(2)
        self.up1 = AttentionUpBlock(features[3] * 2, features[3], features[3])
        self.up2 = AttentionUpBlock(features[3], features[2], features[2])
        self.up3 = AttentionUpBlock(features[2], features[1], features[1])
        self.up4 = AttentionUpBlock(features[1], features[0], features[0])
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bottleneck = self.bottleneck(self.pool(x4))
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)
