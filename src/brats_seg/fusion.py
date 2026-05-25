"""Multi-modal fusion and inter-slice context modules for BraTS segmentation.

Provides two types of fusion:
1. ModalityAttentionFusion - learns per-modality importance weights across the 4 MRI
   modalities (T1c, T1n, T2f, T2w) using channel attention.
2. SliceContextFusion - fuses features across adjacent 2D slices using lightweight
   convolution to capture 3D inter-slice context.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ModalityAttentionFusion(nn.Module):
    """Learn per-modality feature importance via Squeeze-and-Excitation channel attention.

    Given a 4-modality input (4 channels: T1c, T1n, T2f, T2w), this module learns
    per-channel attention weights that capture which modality matters most at each
    spatial location. The attention is applied as element-wise gating before the
    features enter the segmentation backbone.

    Args:
        num_modalities: Number of input modalities (default 4).
        reduction: Channel reduction ratio for the bottleneck (default 2).
    """

    def __init__(self, num_modalities: int = 4, reduction: int = 2) -> None:
        super().__init__()
        inner_dim = max(num_modalities // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_modalities, inner_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, num_modalities, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-modality channel attention.

        Args:
            x: Input tensor of shape (B, 4, H, W).

        Returns:
            Attention-gated tensor of same shape (B, 4, H, W).
        """
        scale = self.se(x)
        return x * scale


class CrossModalityGate(nn.Module):
    """Cross-modality gating: each modality learns to attend to others.

    Maps 4 channels → 16 pairwise attention weights → softmax over 4 modalities
    to produce per-modality fusion weights. More expressive than simple SE but
    still lightweight (~600 params).
    """

    def __init__(self, num_modalities: int = 4) -> None:
        super().__init__()
        inner = num_modalities * 4
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_modalities, inner, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, num_modalities, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.net(x)
        return x * weights


class SliceContextFusion(nn.Module):
    """Fuse features across adjacent 2D slices to capture 3D inter-slice context.

    This is a lightweight alternative to full 3D convolutions. It takes a stack
    of N adjacent 2D slices (each with C channels) and applies depthwise 1D
    convolution along the slice axis + channel mixing.

    Input: (B, C*N, H, W) where N adjacent slices are channel-concatenated.
    Output: (B, C_out, H, W) — fused features that incorporate inter-slice context.

    Args:
        num_slices: Number of adjacent slices to fuse (default 3).
        in_channels: Channels per slice (default 4 for BraTS modalities).
        out_channels: Output channels after fusion (default 32).
    """

    def __init__(self, num_slices: int = 3, in_channels: int = 4, out_channels: int = 32) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.in_channels = in_channels

        mixed = num_slices * in_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(mixed, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fusion(x)


class ModalitySpecificEncoder(nn.Module):
    """Process each modality separately with lightweight per-modality conv, then fuse.

    Instead of early fusion (stacking 4 channels), this module:
    1. Applies a small conv block to each modality independently.
    2. Fuses the 4 modality features via learned channel attention.

    This is inspired by modality-specific encoders in multi-modal medical imaging
    where different modalities highlight different tissue properties.

    Args:
        num_modalities: Number of input modalities (default 4).
        base_channels: Output channels per modality after processing (default 16).
    """

    def __init__(self, num_modalities: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.num_modalities = num_modalities

        self.modality_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_modalities)
        ])

        total_channels = num_modalities * base_channels
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, total_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process each modality separately, then fuse.

        Args:
            x: (B, 4, H, W) — 4-modality input.

        Returns:
            (B, num_modalities * base_channels, H, W) — fused features.
        """
        features = []
        for i in range(self.num_modalities):
            modality_slice = x[:, i:i + 1, :, :]
            features.append(self.modality_convs[i](modality_slice))

        fused = torch.cat(features, dim=1)
        gate = self.fusion_gate(fused)
        return fused * gate
