from __future__ import annotations

import torch
from torch import nn

from .attention_unet import AttentionUpBlock
from .unet import DoubleConv, DownBlock


class ModalityWiseFusion(nn.Module):
    """Fuse same-scale features with sample-wise modality attention."""

    def __init__(self, channels: int, num_modalities: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 1)
        self.prior_logits = nn.Parameter(torch.zeros(num_modalities))
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.cat([self.score(feature) for feature in features], dim=1)
        scores = scores + self.prior_logits.view(1, -1, 1, 1)
        weights = torch.softmax(scores, dim=1)
        fused = sum(feature * weights[:, index : index + 1] for index, feature in enumerate(features))
        return fused, weights.squeeze(-1).squeeze(-1)


class RegionModalityAttentionUNet2D(nn.Module):
    """Attention U-Net with multi-scale modality-wise attention.

    Each input modality is encoded separately. At every encoder scale, a
    sample-wise attention module fuses the modality-specific feature maps before
    the shared Attention U-Net decoder predicts the tumor regions.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.modality_stems = nn.ModuleList(DoubleConv(1, features[0]) for _ in range(in_channels))
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[4])
        self.down5 = DownBlock(features[4], features[5])
        self.fusions = nn.ModuleList(ModalityWiseFusion(channels, in_channels) for channels in features)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[5], features[5] * 2)
        self.up1 = AttentionUpBlock(features[5] * 2, features[5], features[5])
        self.up2 = AttentionUpBlock(features[5], features[4], features[4])
        self.up3 = AttentionUpBlock(features[4], features[3], features[3])
        self.up4 = AttentionUpBlock(features[3], features[2], features[2])
        self.up5 = AttentionUpBlock(features[2], features[1], features[1])
        self.up6 = AttentionUpBlock(features[1], features[0], features[0])
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.register_buffer("attention_ema", torch.full((len(features), in_channels), 1.0 / in_channels))

    def modality_attention(self) -> torch.Tensor:
        attention = self.attention_ema.mean(dim=0)
        return attention.unsqueeze(0).repeat(self.out_channels, 1)

    def _fuse_scale(self, scale_index: int, features: list[torch.Tensor]) -> torch.Tensor:
        fused, weights = self.fusions[scale_index](features)
        self._update_attention_ema(scale_index, weights)
        return fused

    def _update_attention_ema(self, scale_index: int, weights: torch.Tensor) -> None:
        with torch.no_grad():
            batch_attention = weights.detach().mean(dim=0)
            self.attention_ema[scale_index].mul_(0.95).add_(batch_attention, alpha=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [stem(x[:, index : index + 1]) for index, stem in enumerate(self.modality_stems)]
        x1 = self._fuse_scale(0, features)

        features = [self.down1(feature) for feature in features]
        x2 = self._fuse_scale(1, features)

        features = [self.down2(feature) for feature in features]
        x3 = self._fuse_scale(2, features)

        features = [self.down3(feature) for feature in features]
        x4 = self._fuse_scale(3, features)

        features = [self.down4(feature) for feature in features]
        x5 = self._fuse_scale(4, features)

        features = [self.down5(feature) for feature in features]
        x6 = self._fuse_scale(5, features)

        bottleneck = self.bottleneck(self.pool(x6))
        x = self.up1(bottleneck, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.head(x)
