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
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, features: tuple[int, ...] = (8, 16, 32, 64, 128, 256)) -> None:
        super().__init__()
        self.stem = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[4])
        self.down5 = DownBlock(features[4], features[5])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[5], features[5] * 2)
        self.up1 = AttentionUpBlock(features[5] * 2, features[5], features[5])
        self.up2 = AttentionUpBlock(features[5], features[4], features[4])
        self.up3 = AttentionUpBlock(features[4], features[3], features[3])
        self.up4 = AttentionUpBlock(features[3], features[2], features[2])
        self.up5 = AttentionUpBlock(features[2], features[1], features[1])
        self.up6 = AttentionUpBlock(features[1], features[0], features[0])
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        bottleneck = self.bottleneck(self.pool(x6))
        x = self.up1(bottleneck, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.head(x)
