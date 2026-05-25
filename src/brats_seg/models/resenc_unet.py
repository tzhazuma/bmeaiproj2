from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from .unet import DoubleConv


class ResDoubleConv(nn.Module):
    """Double-convolution block with residual skip connection.

    Main path: Conv→BN→ReLU→Conv→BN→ReLU
    Skip path: identity (channels match) or 1×1 conv (channel projection).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.skip(x)


class ResDownBlock(nn.Module):
    """Encoder downsampling block: MaxPool2d(2) + ResDoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder upsampling block: ConvTranspose2d + skip concat + DoubleConv.

    Mirrors the project convention: skip features are concatenated *before*
    the DoubleConv so the conv sees ``out_channels + skip_channels`` inputs.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResEncUNet2D(nn.Module):
    """nnU-Net-inspired 2D U-Net with Residual Encoder blocks and Deep Supervision.

    Architecture::

        stem       ──── ResDoubleConv(4→f0)          ──── skip ──────────► up4
        down1      ──── ResDownBlock(f0→f1)          ──── skip ─────► up3
        down2      ──── ResDownBlock(f1→f2)          ──── skip ► up2
        down3      ──── ResDownBlock(f2→f3)          ── skip ► up1
        down4      ──── ResDownBlock(f3→f4)           ──► bottleneck
        bottleneck ──── ResDoubleConv(f4→f4)          ──► up1 ──► up2 ──► up3 ──► up4 ──► head
                                                        ds↑      ds↑      ds↑
                                                     (aux1)   (aux2)   (aux3)

    Deep-supervision heads at up1/up2/up3 decoder outputs are 1×1 convs
    followed by bilinear upsampling to the input resolution.

    Gradient checkpointing wraps every encoder/bottleneck block (and
    decoder blocks) when ``use_checkpoint=True`` and the model is in
    training mode.

    Parameters
    ----------
    in_channels : int
        Input modality count (default 4 for BraTS).
    out_channels : int
        Output channel count (default 3 for WT/TC/ET).
    features : tuple[int, ...]
        Feature map sizes per U-Net level.  Default ``(32, 64, 128, 256, 320)``
        yields ≈7.5 M parameters — comfortably within 8 GB VRAM at batch_size=2.
    use_checkpoint : bool
        Enable gradient checkpointing on encoder + decoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple[int, ...] = (32, 64, 128, 256, 320),
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        f = features

        # ── Encoder (residual blocks) ────────────────────────────────
        self.stem = ResDoubleConv(in_channels, f[0])       #   4 →  32
        self.down1 = ResDownBlock(f[0], f[1])              #  32 →  64
        self.down2 = ResDownBlock(f[1], f[2])              #  64 → 128
        self.down3 = ResDownBlock(f[2], f[3])              # 128 → 256
        self.down4 = ResDownBlock(f[3], f[4])              # 256 → 320
        self.bottleneck = ResDoubleConv(f[4], f[4])        # 320 → 320

        # ── Decoder ─────────────────────────────────────────────────
        self.up1 = UpBlock(f[4], f[3], f[3])               #  320 + skip 256 → 256
        self.up2 = UpBlock(f[3], f[2], f[2])               #  256 + skip 128 → 128
        self.up3 = UpBlock(f[2], f[1], f[1])               #  128 + skip  64 →  64
        self.up4 = UpBlock(f[1], f[0], f[0])               #   64 + skip  32 →  32

        # ── Output heads ────────────────────────────────────────────
        self.head = nn.Conv2d(f[0], out_channels, kernel_size=1)          # main

        # Deep-supervision auxiliary heads (1×1 conv + bilinear upsample)
        self.ds_up1 = nn.Conv2d(f[3], out_channels, kernel_size=1)        # from up1 out (256 ch)
        self.ds_up2 = nn.Conv2d(f[2], out_channels, kernel_size=1)        # from up2 out (128 ch)
        self.ds_up3 = nn.Conv2d(f[1], out_channels, kernel_size=1)        # from up3 out ( 64 ch)

    # ── helpers ──────────────────────────────────────────────────────

    def _cpt(self, fn: nn.Module, *args: Tensor) -> Tensor:
        """Apply gradient checkpointing when enabled and training."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @staticmethod
    def _ds_upsample(x: Tensor, target_shape: tuple[int, int]) -> Tensor:
        """Bilinear upsample deep-supervision logits to target spatial size."""
        return nn.functional.interpolate(
            x, size=target_shape, mode="bilinear", align_corners=False
        )

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)`` input volume.
        return_aux : bool
            When ``True`` returns ``(main, aux1, aux2, aux3)`` for
            deep-supervision loss computation during training.

        Returns
        -------
        Tensor or tuple of four Tensors
            Main output ``(B, out_channels, H, W)`` (inference) or a
            4-tuple ``(main, aux1, aux2, aux3)`` (training).
        """
        target_shape: tuple[int, int] = x.shape[2:]  # (H, W)

        # ── Encoder ──
        x1 = self._cpt(self.stem, x)          # (B,  32, H,   W  )
        x2 = self._cpt(self.down1, x1)        # (B,  64, H/2, W/2)
        x3 = self._cpt(self.down2, x2)        # (B, 128, H/4, W/4)
        x4 = self._cpt(self.down3, x3)        # (B, 256, H/8, W/8)
        x5 = self._cpt(self.down4, x4)        # (B, 320, H/16,W/16)

        # ── Bottleneck ──
        b = self._cpt(self.bottleneck, x5)    # (B, 320, H/16,W/16)

        # ── Decoder ──
        d1 = self._cpt(self.up1, b, x4)       # (B, 256, H/8, W/8)
        d2 = self._cpt(self.up2, d1, x3)      # (B, 128, H/4, W/4)
        d3 = self._cpt(self.up3, d2, x2)      # (B,  64, H/2, W/2)
        d4 = self._cpt(self.up4, d3, x1)      # (B,  32, H,   W  )

        # ── Main output ──
        main_out = self.head(d4)              # (B, out_channels, H, W)

        if return_aux:
            # Deep-supervision branches (upsampled to input resolution)
            aux1 = self._ds_upsample(self.ds_up1(d1), target_shape)
            aux2 = self._ds_upsample(self.ds_up2(d2), target_shape)
            aux3 = self._ds_upsample(self.ds_up3(d3), target_shape)
            return main_out, aux1, aux2, aux3

        return main_out
