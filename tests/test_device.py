"""Tests for brats_seg.device — device auto-detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_seg import device as dev


class GetDeviceTest(unittest.TestCase):
    """Test get_device() auto-detection logic via monkey-patching torch backends."""

    # ── S1: CUDA (NVIDIA) available → returns "cuda" ─────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_cuda_first_priority(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock, _mock_rocm: mock.Mock) -> None:
        """When CUDA is available (non-ROCm), get_device() returns 'cuda'."""
        d = dev.get_device()
        self.assertEqual(d, "cuda")

    # ── S2: ROCm (AMD) → returns "cuda" (ROCm uses CUDA device string) ───────

    @mock.patch("brats_seg.device._rocm_available", return_value=True)
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_rocm_returns_cuda(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock, _mock_rocm: mock.Mock) -> None:
        """On ROCm, get_device() returns 'cuda' (not 'rocm')."""
        d = dev.get_device()
        self.assertEqual(d, "cuda")

    # ── S3: ROCm device_name() shows AMD ─────────────────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=True)
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.device_count", return_value=1)
    @mock.patch("torch.cuda.get_device_name", return_value="AMD Radeon RX 7900 XTX")
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_rocm_device_name(
        self, _mock_mps: mock.Mock, _mock_name: mock.Mock,
        _mock_cnt: mock.Mock, _mock_cuda: mock.Mock, _mock_rocm: mock.Mock,
    ) -> None:
        """device_name() reports AMD GPU with (ROCm) suffix on ROCm."""
        name = dev.device_name()
        self.assertIn("ROCm", name)
        self.assertIn("AMD", name)

    # ── S4: Only MPS available → returns "mps" ───────────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_fallback(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock, _mock_rocm: mock.Mock) -> None:
        """When only MPS is available, get_device() returns 'mps'."""
        d = dev.get_device()
        self.assertEqual(d, "mps")

    # ── S5: Only XPU available → returns "xpu" ───────────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    @mock.patch("torch.xpu.is_available", return_value=True)
    def test_xpu_fallback(
        self, _mock_xpu: mock.Mock, _mock_mps: mock.Mock,
        _mock_cuda: mock.Mock, _mock_rocm: mock.Mock,
    ) -> None:
        """When only Intel XPU is available, get_device() returns 'xpu'."""
        d = dev.get_device()
        self.assertEqual(d, "xpu")

    # ── S6: No accelerator → CPU fallback ────────────────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    @mock.patch("torch.xpu.is_available", return_value=False)
    def test_cpu_fallback(
        self, _mock_xpu: mock.Mock, _mock_mps: mock.Mock,
        _mock_cuda: mock.Mock, _mock_rocm: mock.Mock,
    ) -> None:
        """When no accelerator is available, get_device() returns 'cpu'."""
        d = dev.get_device()
        self.assertEqual(d, "cpu")

    # ── S7: device_name() returns human-readable string ──────────────────────

    def test_device_name_returns_string(self) -> None:
        """device_name() always returns a non-empty string."""
        name = dev.device_name()
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    # ── S8: device_type() returns "gpu" for CUDA ─────────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_device_type_gpu(
        self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock, _mock_rocm: mock.Mock,
    ) -> None:
        """device_type() returns 'gpu' for CUDA."""
        self.assertEqual(dev.device_type(), "gpu")

    # ── S9: device_type() returns "cpu" when no GPU ──────────────────────────

    @mock.patch("brats_seg.device._rocm_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    @mock.patch("torch.xpu.is_available", return_value=False)
    def test_device_type_cpu(
        self, _mock_xpu: mock.Mock, _mock_mps: mock.Mock,
        _mock_cuda: mock.Mock, _mock_rocm: mock.Mock,
    ) -> None:
        """device_type() returns 'cpu' when no GPU."""
        self.assertEqual(dev.device_type(), "cpu")

    # ── S10: get_device() re-import consistency ──────────────────────────────

    def test_get_device_importable_from_package(self) -> None:
        """get_device is accessible from brats_seg directly."""
        from brats_seg import get_device as gd  # type: ignore[attr-defined]

        self.assertIs(gd, dev.get_device)


if __name__ == "__main__":
    unittest.main()
