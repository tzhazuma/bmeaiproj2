from __future__ import annotations

import torch
from torch import Tensor, nn

from .unet import DoubleConv, DownBlock


class MultiScaleConv(nn.Module):
    """Parallel dilated convolutions (rates 1, 2, 4) fused via 1x1 conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, c3, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch4(x)], dim=1)
        return self.fuse(out)


class ChannelGate(nn.Module):
    """SE-style squeeze-and-excitation channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.fc(self.gap(x))
        return x * weight


class SpatialGate(nn.Module):
    """1x1 conv + sigmoid spatial attention (self-gating, distinct from AttentionGate)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class MAUBlock(nn.Module):
    """DoubleConv + ChannelGate + SpatialGate combined decoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.channel_gate = ChannelGate(out_channels)
        self.spatial_gate = SpatialGate(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


class _AuxHead(nn.Module):
    """Lightweight auxiliary head for deep supervision."""

    def __init__(self, in_channels: int, scale_factor: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MAUUpBlock(nn.Module):
    """Decoder block: upsample → channel-attend skip → MAUBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.channel_gate = ChannelGate(skip_channels)
        self.mau = MAUBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        x = nn.functional.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        skip = self.channel_gate(skip)
        return self.mau(torch.cat([skip, x], dim=1))


class MAUNet2D(nn.Module):
    """Multi-scale Attention U-Net for medical image segmentation.

    Key features:
    - Multi-scale dilated convolutions at the bottleneck
    - Channel attention on each skip connection
    - Spatial self-attention at the bottleneck
    - Deep supervision via three auxiliary outputs

    Args:
        in_channels: Number of input modalities (default 4 for BraTS).
        out_channels: Number of output tumor regions (default 3: WT/TC/ET).
        features: Channel widths per encoder level.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        f = features

        # ── Encoder ──────────────────────────────────────────────────
        self.stem = DoubleConv(in_channels, f[0])
        self.down1 = DownBlock(f[0], f[1])
        self.down2 = DownBlock(f[1], f[2])
        self.down3 = DownBlock(f[2], f[3])
        self.pool = nn.MaxPool2d(2)

        # ── Multi-scale bottleneck ───────────────────────────────────
        self.bottleneck_conv = MultiScaleConv(f[3], f[3] * 2)
        self.bottleneck_spatial = SpatialGate(f[3] * 2)

        # ── Decoder ──────────────────────────────────────────────────
        self.up1 = MAUUpBlock(f[3] * 2, f[3], f[3])
        self.up2 = MAUUpBlock(f[3], f[2], f[2])
        self.up3 = MAUUpBlock(f[2], f[1], f[1])
        self.up4 = MAUUpBlock(f[1], f[0], f[0])

        self.head = nn.Conv2d(f[0], out_channels, kernel_size=1)

        # ── Deep supervision heads ───────────────────────────────────
        self.aux3_head = _AuxHead(f[3], scale_factor=8.0)
        self.aux2_head = _AuxHead(f[2], scale_factor=4.0)
        self.aux1_head = _AuxHead(f[1], scale_factor=2.0)

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        # Encoder
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        bottleneck = self.bottleneck_spatial(self.bottleneck_conv(self.pool(x4)))

        # Decoder
        dec_up1 = self.up1(bottleneck, x4)
        dec_up2 = self.up2(dec_up1, x3)
        dec_up3 = self.up3(dec_up2, x2)
        dec_up4 = self.up4(dec_up3, x1)

        main = self.head(dec_up4)

        if return_aux:
            aux3 = self.aux3_head(dec_up1)
            aux2 = self.aux2_head(dec_up2)
            aux1 = self.aux1_head(dec_up3)
            return main, aux1, aux2, aux3

        return main
