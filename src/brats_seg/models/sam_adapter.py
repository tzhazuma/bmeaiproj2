"""SAM-based medical image segmentation fine-tuning for BraTS.

Provides:

- ``SAMAdapter``: Wraps SAM (Segment Anything Model) as a segmentation
  backbone for BraTS.  Supports loading pretrained SAM encoders and
  attaching a lightweight decoder head for 3-region (WT/TC/ET) output.
- ``SAMAdapterConfig``: Dataclass for configuration.
- ``create_sam_adapter``: Factory function to instantiate from config.

The module works in **two modes**:

1. **Full SAM mode** (``sam`` source) — uses ``segment_anything``.
   Requires ``pip install segment-anything`` and the pretrained checkpoint.
2. **Standalone mode** (``vit`` source) — a pure-PyTorch ViT-inspired
   encoder with random init.  Useful for testing, architecture
   exploration, or when SAM weights are unavailable.

Usage::

    from brats_seg.models.sam_adapter import SAMAdapterConfig, create_sam_adapter

    cfg = SAMAdapterConfig(source="vit", freeze_encoder=True)
    model = create_sam_adapter(cfg)
    logits = model(torch.randn(2, 4, 160, 160))   # (B, 3, H, W)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SAMAdapterConfig:
    """Configuration for SAMAdapter creation.

    Attributes:
        source:
            ``"sam"``  — load a real SAM model (requires ``segment-anything`` +
                        checkpoint).
            ``"vit"``  — use a pure-PyTorch ViT-inspired encoder (no external
                        dependency, random init).
        model_type:
            SAM model variant: ``"vit_b"``, ``"vit_l"``, or ``"vit_h"``.
            Only used when ``source="sam"``.
        checkpoint_path:
            Path to the SAM checkpoint ``.pth`` file (downloaded from Meta).
            Required when ``source="sam"``.
        in_channels:
            Number of input channels (default 4 for BraTS modalities).
        out_channels:
            Number of output channels (default 3 for WT/TC/ET).
        img_size:
            Input image size (assumed square, default 160 for BraTS slices).
        patch_size:
            ViT patch size (default 16).
        encoder_embed_dim:
            ViT embedding dimension (default 768 for vit_b).
        encoder_depth:
            Number of ViT transformer blocks (default 12 for vit_b).
        encoder_num_heads:
            Number of attention heads (default 12 for vit_b).
        freeze_encoder:
            If True, the image encoder weights are frozen during training.
        decoder_dim:
            Hidden dimension of the task-specific decoder head.
    """

    source: Literal["sam", "vit"] = "vit"
    model_type: Literal["vit_b", "vit_l", "vit_h"] = "vit_b"
    checkpoint_path: str = ""
    in_channels: int = 4
    out_channels: int = 3
    img_size: int = 160
    patch_size: int = 16
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    freeze_encoder: bool = True
    decoder_dim: int = 256

    def __post_init__(self) -> None:
        if self.source == "sam" and not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required when source='sam'. "
                "Download from https://github.com/facebookresearch/segment-anything"
            )


# ── Pure-PyTorch ViT Encoder (standalone mode) ──────────────────────────────

class PatchEmbed(nn.Module):
    """2D image to patch embedding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)  # (B, embed_dim, H/p, W/p)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (ViT-B/16) used as the SAM-like image encoder.

    This is a pure-PyTorch implementation *inspired* by SAM's image encoder
    architecture but not an exact replica.  It serves as a lightweight
    backbone when the real SAM weights are unavailable.
    """

    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 160,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, D, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        # Interpolate positional embedding for non-native input sizes
        pos_embed = self.pos_embed
        num_patches = Hp * Wp
        N_pos = pos_embed.shape[1]
        if N_pos != num_patches:
            D = pos_embed.shape[-1]
            grid = int(math.isqrt(N_pos))
            pos_embed = pos_embed.transpose(1, 2).reshape(1, D, grid, grid)
            pos_embed = nn.functional.interpolate(pos_embed, size=(Hp, Wp), mode="bilinear", align_corners=False)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        return x  # (B, embed_dim, Hp, Wp)


# ── Lightweight Decoder Head ─────────────────────────────────────────────────

class SAMDecoderHead(nn.Module):
    """Task-specific decoder head that maps SAM image embeddings to segmentation.

    Architecture: 2× conv upsampling layers + output projection.
    """

    def __init__(self, in_channels: int, decoder_dim: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, decoder_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(decoder_dim)
        self.conv2 = nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(decoder_dim)
        self.output_proj = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.output_proj(x)


# ── SAM Adapter ──────────────────────────────────────────────────────────────

class SAMAdapter(nn.Module):
    """SAM-based segmentation model adapted for BraTS fine-tuning.

    Wraps SAM's image encoder (or a pure-PyTorch ViT) as a frozen or
    trainable backbone, and attaches a lightweight decoder head that
    produces ``out_channels`` segmentation maps.

    Args:
        config: :class:`SAMAdapterConfig` instance.
    """

    def __init__(self, config: SAMAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.source = config.source

        # ── Build / load encoder ─────────────────────────────────────────
        if config.source == "sam":
            self._init_sam_encoder()
        else:
            self._init_vit_encoder()

        # ── Freeze encoder if requested ──────────────────────────────────
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ── Decoder head ─────────────────────────────────────────────────
        embed_dim = config.encoder_embed_dim
        self.decoder = SAMDecoderHead(embed_dim, config.decoder_dim, config.out_channels)

    def _init_vit_encoder(self) -> None:
        self.encoder = ViTEncoder(
            in_channels=self.config.in_channels,
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.encoder_embed_dim,
            depth=self.config.encoder_depth,
            num_heads=self.config.encoder_num_heads,
        )

    def _init_sam_encoder(self) -> None:
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "source='sam' requires 'segment-anything'. "
                "Install with: pip install segment-anything\n"
                "Alternatively, use source='vit' for a standalone mode."
            )

        sam = sam_model_registry[self.config.model_type](checkpoint=self.config.checkpoint_path)
        # Use SAM's image encoder; discard prompt encoder and mask decoder.
        if self.config.in_channels != 3:
            old_conv = sam.image_encoder.patch_embed.proj
            new_conv = nn.Conv2d(
                self.config.in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            sam.image_encoder.patch_embed.proj = new_conv

        self.encoder = sam.image_encoder
        self.config.encoder_embed_dim = self.encoder.pos_embed.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Logits of shape ``(B, out_channels, H, W)``.
        """
        H, W = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        # Resize to match input spatial dimensions
        if logits.shape[-2:] != (H, W):
            logits = nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# ── Factory function ─────────────────────────────────────────────────────────

def create_sam_adapter(config: SAMAdapterConfig | None = None, **kwargs) -> SAMAdapter:
    """Create a SAMAdapter from a config or keyword arguments.

    Args:
        config: Optional :class:`SAMAdapterConfig`.
        **kwargs: Override or supply config fields when *config* is ``None``.

    Returns:
        A :class:`SAMAdapter` instance.
    """
    if config is None:
        config = SAMAdapterConfig(**kwargs)
    elif kwargs:
        raise ValueError("Provide either config object or keyword arguments, not both.")
    return SAMAdapter(config)
