from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .unet import DoubleConv, UpBlock


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition ``(B, H, W, C)`` into non-overlapping windows.

    Returns ``(B * num_windows, window_size, window_size, C)``.
    """
    B, H, W, C = x.shape
    nW_h = H // window_size
    nW_w = W // window_size
    x = x.view(B, nW_h, window_size, nW_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(
    windows: Tensor, window_size: int, H: int, W: int
) -> Tensor:
    """Reverse :func:`window_partition` back to ``(B, H, W, C)``."""
    nW_total = windows.shape[0]
    nW_h = H // window_size
    nW_w = W // window_size
    B = nW_total // (nW_h * nW_w)
    x = windows.view(B, nW_h, nW_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Swin Transformer building blocks
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Operates on token sequences of shape ``(N, C)`` where *N* is the number
    of patches inside a single window (``window_size²``).
    """

    def __init__(
        self, dim: int, window_size: int, num_heads: int
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # --- relative position bias table ---
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        # --- pre-compute index LUT ---
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        )  # (2, ws, ws)
        coords = coords.reshape(2, -1)  # (2, ws*ws)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0)  # (ws*ws, ws*ws, 2)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        """*x*: ``(B_, N, C)``  where ``N = window_size²``."""
        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        # relative position bias
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        rpb = rpb.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        # shifted-window attention mask
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinMLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    """One Swin Transformer block — either W-MSA or SW-MSA.

    Works in ``(B, H, W, C)`` channel-last layout.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim, mlp_ratio)

    # ------------------------------------------------------------------
    # attention mask for shifted windows
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mask(
        H: int, W: int, window_size: int, shift_size: int, device: torch.device
    ) -> Tensor | None:
        if shift_size == 0:
            return None

        ws = window_size
        ss = shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)

        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        return attn_mask.masked_fill(attn_mask == 0, 0.0)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """*x*: ``(B, H, W, C)``.  Returns same shape."""
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        # ---------- pad to be divisible by window_size ----------
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        Hp, Wp = x.shape[1], x.shape[2]

        # ---------- cyclic shift (SW-MSA) ----------
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # ---------- window partition ----------
        x_w = window_partition(x, ws)  # (B*nW, ws, ws, C)
        x_w = x_w.view(-1, ws * ws, C)

        # ---------- attention ----------
        attn_mask = self._build_mask(Hp, Wp, ws, self.shift_size, x.device)
        attn_out = self.attn(x_w, attn_mask)

        # ---------- window reverse ----------
        attn_out = attn_out.view(-1, ws, ws, C)
        x = window_reverse(attn_out, ws, Hp, Wp)

        # ---------- reverse cyclic shift ----------
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )

        # ---------- remove padding ----------
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        # residual
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """Halve spatial resolution and double channels via 2×2 patch merging.

    Input:  ``(B,  H,  W,    C)``
    Output: ``(B, H/2, W/2, 2C)``
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        # regroup 2×2 patches
        x = x.reshape(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
        x = x.reshape(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        return self.reduction(x)


# ---------------------------------------------------------------------------
# SwinUNETR2D
# ---------------------------------------------------------------------------


class SwinUNETR2D(nn.Module):
    """2D Swin Transformer encoder + CNN decoder for medical segmentation.

    Parameters
    ----------
    in_channels:
        Input modality count (default 4 for BraTS).
    out_channels:
        Number of segmentation classes (default 3).
    init_dim:
        Base feature dimension after patch embedding.  Smaller than the
        canonical 96 to save VRAM.
    use_checkpoint:
        Enable gradient checkpointing on every Swin block.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        init_dim: int = 48,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if init_dim % 3 != 0:
            raise ValueError(
                f"init_dim ({init_dim}) must be divisible by 3 "
                f"(head count of stage 1)"
            )

        # ---- Encoder ----

        # Patch embedding: Conv2d -> (B, init_dim, 40, 40)
        self.patch_embed = nn.Conv2d(
            in_channels, init_dim, kernel_size=4, stride=4
        )

        dim1 = init_dim  # 48
        dim2 = init_dim * 2  # 96
        dim3 = init_dim * 4  # 192
        dim_bn = init_dim * 8  # 384

        # Stage 1 — 2 blocks + PatchMerging
        self.stage1_blocks = nn.ModuleList([
            SwinTransformerBlock(dim1, window_size=8, num_heads=3, shift_size=0),
            SwinTransformerBlock(dim1, window_size=8, num_heads=3, shift_size=4),
        ])
        self.stage1_merge = PatchMerging(dim1)

        # Stage 2 — 2 blocks + PatchMerging
        self.stage2_blocks = nn.ModuleList([
            SwinTransformerBlock(dim2, window_size=8, num_heads=6, shift_size=0),
            SwinTransformerBlock(dim2, window_size=8, num_heads=6, shift_size=4),
        ])
        self.stage2_merge = PatchMerging(dim2)

        # Stage 3 — 2 blocks + PatchMerging
        self.stage3_blocks = nn.ModuleList([
            SwinTransformerBlock(dim3, window_size=5, num_heads=12, shift_size=0),
            SwinTransformerBlock(dim3, window_size=5, num_heads=12, shift_size=2),
        ])
        self.stage3_merge = PatchMerging(dim3)

        # Bottleneck — 2 blocks, no downsampling
        self.bottleneck_blocks = nn.ModuleList([
            SwinTransformerBlock(dim_bn, window_size=5, num_heads=24, shift_size=0),
            SwinTransformerBlock(dim_bn, window_size=5, num_heads=24, shift_size=2),
        ])

        # ---- Decoder (channel-first, (B, C, H, W)) ----

        self.decoder3 = UpBlock(dim_bn, dim3, dim3)  # 384→192  skip=192
        self.decoder2 = UpBlock(dim3, dim2, dim2)  # 192→96   skip=96
        self.decoder1 = UpBlock(dim2, dim1, dim1)  #  96→48   skip=48

        # Level 0 — stride-4 up + skip from raw input
        self.decoder0_up = nn.ConvTranspose2d(
            dim1, dim1, kernel_size=4, stride=4
        )
        self.decoder0_conv = DoubleConv(dim1 + in_channels, dim1)

        self.head = nn.Conv2d(dim1, out_channels, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    # weight init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chw_to_hwc(x: Tensor) -> Tensor:
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _hwc_to_chw(x: Tensor) -> Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _forward_blocks(
        self, blocks: nn.ModuleList, x: Tensor
    ) -> Tensor:
        """Run *blocks* sequentially, checkpointing each if enabled."""
        for blk in blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """*x*: ``(B, 4, 160, 160)`` → ``(B, 3, 160, 160)``."""
        x_input = x  # (B, 4, 160, 160)

        # patch embedding → channel-last
        x = self.patch_embed(x)  # (B,  48,  40,  40)
        x = self._chw_to_hwc(x)  # (B,  40,  40,  48)

        # ---- Encoder ----

        x = self._forward_blocks(self.stage1_blocks, x)  # (B,  40,  40,  48)
        skip1 = x
        x = self.stage1_merge(x)  # (B,  20,  20,  96)

        x = self._forward_blocks(self.stage2_blocks, x)  # (B,  20,  20,  96)
        skip2 = x
        x = self.stage2_merge(x)  # (B,  10,  10, 192)

        x = self._forward_blocks(self.stage3_blocks, x)  # (B,  10,  10, 192)
        skip3 = x
        x = self.stage3_merge(x)  # (B,   5,   5, 384)

        x = self._forward_blocks(self.bottleneck_blocks, x)  # (B,  5,  5, 384)

        # convert to channel-first for CNN decoder
        x = self._hwc_to_chw(x)  # (B, 384,   5,   5)
        skip1 = self._hwc_to_chw(skip1)  # (B,  48,  40,  40)
        skip2 = self._hwc_to_chw(skip2)  # (B,  96,  20,  20)
        skip3 = self._hwc_to_chw(skip3)  # (B, 192,  10,  10)

        # ---- Decoder ----

        x = self.decoder3(x, skip3)  # (B, 192,  10,  10)
        x = self.decoder2(x, skip2)  # (B,  96,  20,  20)
        x = self.decoder1(x, skip1)  # (B,  48,  40,  40)

        # Level 0
        x = self.decoder0_up(x)  # (B,  48, 160, 160)
        x = torch.cat([x_input, x], dim=1)  # (B,  52, 160, 160)
        x = self.decoder0_conv(x)  # (B,  48, 160, 160)

        return self.head(x)  # (B,  3, 160, 160)
