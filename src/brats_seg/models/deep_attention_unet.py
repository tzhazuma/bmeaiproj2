from __future__ import annotations

from torch import Tensor, nn

from .attention_unet import AttentionUpBlock
from .unet import DoubleConv, DownBlock


class _AuxHead(nn.Module):

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


class DeepSupAttentionUNet2D(nn.Module):
    """Attention U-Net with deep supervision.

    Three auxiliary heads after ``up1``, ``up2``, ``up3`` upsample
    intermediate decoder features to full resolution.  Training with
    ``return_aux=True``: combined loss = main + 0.5×aux1 + 0.3×aux2 + 0.2×aux3.
    Inference with ``return_aux=False``: main prediction only.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    ) -> None:
        super().__init__()

        self.stem = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[4])
        self.down5 = DownBlock(features[4], features[5])
        self.bottleneck = DoubleConv(features[5], features[5] * 2)

        self.up1 = AttentionUpBlock(features[5] * 2, features[5], features[5])
        self.up2 = AttentionUpBlock(features[5], features[4], features[4])
        self.up3 = AttentionUpBlock(features[4], features[3], features[3])
        self.up4 = AttentionUpBlock(features[3], features[2], features[2])
        self.up5 = AttentionUpBlock(features[2], features[1], features[1])
        self.up6 = AttentionUpBlock(features[1], features[0], features[0])

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.aux3_head = _AuxHead(features[3], scale_factor=8.0)
        self.aux2_head = _AuxHead(features[2], scale_factor=4.0)
        self.aux1_head = _AuxHead(features[1], scale_factor=2.0)

    def forward(self, x: Tensor, return_aux: bool = False) -> (
        Tensor | tuple[Tensor, Tensor, Tensor, Tensor]
    ):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(x6))

        dec_up1 = self.up1(bottleneck, x6)
        dec_up2 = self.up2(dec_up1, x5)
        dec_up3 = self.up3(dec_up2, x4)
        dec_up4 = self.up4(dec_up3, x3)
        dec_up5 = self.up5(dec_up4, x2)
        dec_up6 = self.up6(dec_up5, x1)

        main = self.head(dec_up6)

        if return_aux:
            aux3 = self.aux3_head(dec_up3)
            aux2 = self.aux2_head(dec_up4)
            aux1 = self.aux1_head(dec_up5)
            return main, aux1, aux2, aux3

        return main
