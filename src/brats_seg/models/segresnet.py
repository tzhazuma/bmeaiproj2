from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _num_groups(channels: int) -> int:
    """GroupNorm groups: min(channels//4, 8) with floor of 1."""
    return max(min(channels // 4, 8), 1)


class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm and spatial dropout."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        groups = _num_groups(channels)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.gn1(x))
        out = self.conv1(out)
        out = self.relu(self.gn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual


class DownResBlock(nn.Module):
    """Residual block with stride-2 downsampling and optional channel change."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        groups_in = _num_groups(in_channels)
        groups_out = _num_groups(out_channels)

        self.gn1 = nn.GroupNorm(groups_in, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        out = self.relu(self.gn1(x))
        out = self.conv1(out)
        out = self.relu(self.gn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual


class UpBlock(nn.Module):
    """Upsampling block with skip connection and residual processing."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        merged_ch = out_channels + skip_channels
        groups_merge = _num_groups(merged_ch)
        self.merge = nn.Sequential(
            nn.GroupNorm(groups_merge, merged_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(merged_ch, out_channels, kernel_size=3, padding=1, bias=False),
        )

        self.res_block = ResBlock(out_channels, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        x = self.res_block(x)
        return x


class VAEBranch(nn.Module):
    """VAE regularization branch: encodes bottleneck features, samples latent z, reconstructs input MRI."""

    def __init__(
        self,
        in_channels: int = 256,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_encode = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
        )

        self.recon_decoder = nn.ModuleDict({
            "up0": nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.GroupNorm(_num_groups(128), 128),
                nn.ReLU(inplace=True),
            ),
            "up1": nn.Sequential(
                nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_num_groups(64), 64),
                nn.ReLU(inplace=True),
            ),
            "up2": nn.Sequential(
                nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_num_groups(32), 32),
                nn.ReLU(inplace=True),
            ),
            "up3": nn.Sequential(
                nn.Upsample(scale_factor=5.0, mode="bilinear", align_corners=False),
                nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_num_groups(16), 16),
                nn.ReLU(inplace=True),
            ),
            "up4": nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ),
            "head": nn.Conv2d(16, 4, kernel_size=1),
        })

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = x.shape[0]

        h = self.pool(x).view(B, -1)
        h = self.fc_encode(h)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        z = self.fc_decode(z).view(B, 256, 1, 1)

        recon = self.recon_decoder["up0"](z)
        recon = self.recon_decoder["up1"](recon)
        recon = self.recon_decoder["up2"](recon)
        recon = self.recon_decoder["up3"](recon)
        recon = self.recon_decoder["up4"](recon)
        recon = self.recon_decoder["head"](recon)

        return recon, mu, log_var


class SegResNet2D(nn.Module):
    """Lightweight 2D SegResNet for brain tumor segmentation with VAE regularization.

    Asymmetric encoder-decoder with GroupNorm, spatial dropout, deep supervision,
    and a VAE branch at the bottleneck for input reconstruction.
    Defaults: 4-channel MRI input, 3-class tumor sub-region output.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        init_filters: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        f = init_filters
        ch = [f, f * 2, f * 4, f * 8]

        # ---- Encoder ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(ch[0]), ch[0]),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DownResBlock(ch[0], ch[1], stride=2, dropout=dropout)
        self.enc2 = DownResBlock(ch[1], ch[2], stride=2, dropout=dropout)
        self.enc3 = DownResBlock(ch[2], ch[3], stride=2, dropout=dropout)
        self.bottleneck = DownResBlock(ch[3], ch[3], stride=2, dropout=dropout)

        # ---- Decoder ----
        self.dec3 = UpBlock(ch[3], ch[3], ch[2], dropout=dropout)  # skip: enc3
        self.dec2 = UpBlock(ch[2], ch[2], ch[1], dropout=dropout)  # skip: enc2
        self.dec1 = UpBlock(ch[1], ch[1], ch[0], dropout=dropout)  # skip: enc1
        self.dec0 = UpBlock(ch[0], ch[0], ch[0], dropout=dropout)  # skip: stem

        # ---- Heads ----
        self.head = nn.Conv2d(ch[0], out_channels, kernel_size=1)
        self.aux_head1 = nn.Conv2d(ch[2], out_channels, kernel_size=1)   # from dec3
        self.aux_head2 = nn.Conv2d(ch[1], out_channels, kernel_size=1)   # from dec2

        # ---- VAE Branch ----
        self.vae = VAEBranch(in_channels=ch[3], latent_dim=128)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.bottleneck(s3)

        d3 = self.dec3(bottleneck, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        main_out = self.head(d0)

        if not self.training:
            return main_out

        target_size = x.shape[2:]
        aux1 = F.interpolate(self.aux_head1(d3), size=target_size, mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux_head2(d2), size=target_size, mode="bilinear", align_corners=False)

        vae_recon, mu, log_var = self.vae(bottleneck)
        if vae_recon.shape[2:] != target_size:
            vae_recon = F.interpolate(vae_recon, size=target_size, mode="bilinear", align_corners=False)

        return main_out, aux1, aux2, vae_recon, mu, log_var
