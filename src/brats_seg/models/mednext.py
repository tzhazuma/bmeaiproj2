from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    """LayerNorm applied to the channel dimension via NHWC permute."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: LayerNorm → Depthwise Conv7x7 → LayerNorm → 1×1 expand →
    GELU → 1×1 compress + residual connection.

    Supports downsampling (stride=2) with channel change and gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        expand_ratio: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        mid_channels = out_channels * expand_ratio
        self.use_checkpoint = use_checkpoint

        self.ln1 = LayerNorm2d(in_channels)
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )

        self.ln2 = LayerNorm2d(in_channels)
        self.pw_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False,
            )
        else:
            self.residual = nn.Identity()

    def _forward_impl(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        x = self.ln1(x)
        x = self.dw_conv(x)
        x = self.ln2(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        return x + residual

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class MedNeXtEncoderStage(nn.Module):
    """Encoder stage: N ConvNeXt blocks. First block optionally downsamples."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        kernel_size: int = 7,
        expand_ratio: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            c_in = in_channels if i == 0 else out_channels
            s = 2 if i == 0 else 1
            blocks.append(
                ConvNeXtBlock(c_in, out_channels, kernel_size, stride=s,
                              expand_ratio=expand_ratio, use_checkpoint=use_checkpoint)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class MedNeXtDecoderStage(nn.Module):
    """Decoder stage: ConvTranspose2d upsample → concat skip → ConvNeXtBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        expand_ratio: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvNeXtBlock(
            out_channels + skip_channels, out_channels,
            kernel_size, stride=1, expand_ratio=expand_ratio,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MedNeXt2D(nn.Module):
    """MedNeXt2D — ConvNeXt-based U-Net for 2D medical image segmentation.

    Architecture:
        Stem (conv7×7 + LayerNorm) → Encoder stages (32→64→128→256)
        → Decoder stages (256→128→64→32) → class head (Conv2d 1×1)

    Default Light variant: expand_ratio=2, 3 encoder stages, ~3.2M params.

    Args:
        in_channels:  Input channels (default 4 for BraTS multimodal).
        out_channels: Output channels (default 3 for tumor subregions).
        kernel_size:  Depthwise convolution kernel size.
        encoder_channels: Channel sequence [stem_out, enc1, enc2, enc3].
        expand_ratio: Bottleneck expansion factor (2=Light, 4=regular).
        enc_blocks_per_stage: ConvNeXt blocks per encoder stage.
        use_checkpoint: Wrap each ConvNeXtBlock with gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        kernel_size: int = 7,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256),
        expand_ratio: int = 2,
        enc_blocks_per_stage: int = 2,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        enc_chs = encoder_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_chs[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            LayerNorm2d(enc_chs[0]),
        )

        self.enc_stages = nn.ModuleList()
        for i in range(len(enc_chs) - 1):
            self.enc_stages.append(
                MedNeXtEncoderStage(
                    in_channels=enc_chs[i],
                    out_channels=enc_chs[i + 1],
                    num_blocks=enc_blocks_per_stage,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.dec_stages = nn.ModuleList()
        for j in range(len(enc_chs) - 1):
            in_ch = enc_chs[-(j + 1)]
            out_ch = enc_chs[-(j + 2)]
            skip_ch = enc_chs[-(j + 2)]
            self.dec_stages.append(
                MedNeXtDecoderStage(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.head = nn.Conv2d(enc_chs[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        skips: list[Tensor] = [x]
        for stage in self.enc_stages:
            x = stage(x)
            skips.append(x)

        for i, stage in enumerate(self.dec_stages):
            x = stage(x, skips[-(i + 2)])

        return self.head(x)
