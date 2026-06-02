"""Tests for brats_seg.models.sam_adapter — SAM-like fine-tuning module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_seg.models.sam_adapter import (
    SAMAdapter,
    SAMAdapterConfig,
    create_sam_adapter,
)


class SAMAdapterTest(unittest.TestCase):
    """Test SAMAdapter in standalone (vit) mode — no segment-anything dependency."""

    B = 2
    C_IN = 4
    C_OUT = 3
    H = W = 160

    def _random_input(self) -> torch.Tensor:
        return torch.randn(self.B, self.C_IN, self.H, self.W)

    # ── S1: Forward pass returns correct shape (vit mode) ────────────────────

    def test_forward_output_shape_vit(self) -> None:
        """SAMAdapter(vit) forward returns (B, 3, H, W)."""
        model = create_sam_adapter(source="vit", in_channels=4, out_channels=3)
        x = self._random_input()
        logits = model(x)
        self.assertEqual(logits.shape, (self.B, self.C_OUT, self.H, self.W))

    # ── S2: Forward with small image preserves output shape ──────────────────

    def test_forward_small_image(self) -> None:
        """SAMAdapter handles non-standard input sizes via interpolation."""
        model = create_sam_adapter(source="vit", in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 64, 64)
        logits = model(x)
        self.assertEqual(logits.shape, (1, 3, 64, 64))

    # ── S3: freeze_encoder prevents gradient flow to encoder ─────────────────

    def test_freeze_encoder_disables_grad(self) -> None:
        """When freeze_encoder=True, encoder params have requires_grad=False."""
        model = create_sam_adapter(source="vit", freeze_encoder=True, in_channels=4)
        for name, param in model.encoder.named_parameters():
            self.assertFalse(
                param.requires_grad,
                msg=f"Encoder parameter '{name}' should be frozen",
            )

    # ── S4: trainable encoder has requires_grad=True ─────────────────────────

    def test_unfrozen_encoder_has_grad(self) -> None:
        """When freeze_encoder=False, encoder params have requires_grad=True."""
        model = create_sam_adapter(source="vit", freeze_encoder=False, in_channels=4)
        encoder_has_trainable = any(param.requires_grad for param in model.encoder.parameters())
        self.assertTrue(encoder_has_trainable)

    # ── S5: Decoder parameters are always trainable ──────────────────────────

    def test_decoder_always_trainable(self) -> None:
        """Decoder head parameters are always trainable regardless of freeze_encoder."""
        model = create_sam_adapter(source="vit", freeze_encoder=True, in_channels=4)
        for name, param in model.decoder.named_parameters():
            self.assertTrue(
                param.requires_grad,
                msg=f"Decoder parameter '{name}' should be trainable",
            )

    # ── S6: create_sam_adapter with config object ────────────────────────────

    def test_create_with_config_object(self) -> None:
        """create_sam_adapter accepts a SAMAdapterConfig object."""
        cfg = SAMAdapterConfig(source="vit", in_channels=4, out_channels=3)
        model = create_sam_adapter(cfg)
        self.assertIsInstance(model, SAMAdapter)
        x = self._random_input()
        logits = model(x)
        self.assertEqual(logits.shape, (self.B, self.C_OUT, self.H, self.W))

    # ── S7: gradient flows through entire model ──────────────────────────────

    def test_gradient_flow(self) -> None:
        """Backward pass succeeds and produces gradients for decoder."""
        model = create_sam_adapter(source="vit", freeze_encoder=True, in_channels=4, out_channels=3)
        x = self._random_input()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        has_decoder_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.decoder.parameters()
        )
        self.assertTrue(has_decoder_grad)

    # ── S8: SAMAdapter importable from models package ────────────────────────

    def test_importable_from_models(self) -> None:
        """SAMAdapter is accessible from brats_seg.models."""
        from brats_seg.models import SAMAdapter as SA  # type: ignore[attr-defined]

        self.assertIs(SA, SAMAdapter)


class SAMAdapterConfigTest(unittest.TestCase):
    """Test SAMAdapterConfig validation."""

    def test_sam_source_requires_checkpoint(self) -> None:
        """source='sam' without checkpoint_path raises ValueError."""
        with self.assertRaises(ValueError):
            SAMAdapterConfig(source="sam")

    def test_vit_source_no_checkpoint_needed(self) -> None:
        """source='vit' does not require a checkpoint_path."""
        cfg = SAMAdapterConfig(source="vit")
        self.assertEqual(cfg.source, "vit")


if __name__ == "__main__":
    unittest.main()
