from __future__ import annotations

import torch
from torch import nn

from .attention_unet import AttentionUNet2D


class RegionModalityAttentionUNet2D(nn.Module):
    """Attention U-Net with region-specific modality attention.

    Each output region learns a separate softmax distribution over the input
    modalities before a dedicated Attention U-Net branch predicts that region.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.modality_logits = nn.Parameter(torch.zeros(out_channels, in_channels))
        self.region_heads = nn.ModuleList(
            AttentionUNet2D(in_channels=in_channels, out_channels=1, features=features)
            for _ in range(out_channels)
        )

    def modality_attention(self) -> torch.Tensor:
        return torch.softmax(self.modality_logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.modality_attention()
        logits = []
        for region_index, head in enumerate(self.region_heads):
            weights = attention[region_index].view(1, x.size(1), 1, 1)
            logits.append(head(x * weights))
        return torch.cat(logits, dim=1)
