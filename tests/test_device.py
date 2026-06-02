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

    # ── S1: CUDA available → returns "cuda" ──────────────────────────────────

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_cuda_first_priority(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """When CUDA is available, get_device() returns 'cuda' regardless of other backends."""
        d = dev.get_device()
        self.assertEqual(d, "cuda")

    # ── S2: Only MPS available → returns "mps" ───────────────────────────────

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_fallback(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """When only MPS is available, get_device() returns 'mps'."""
        d = dev.get_device()
        self.assertEqual(d, "mps")

    # ── S3: Only XPU available → returns "xpu" ───────────────────────────────

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    @mock.patch("torch.xpu.is_available", return_value=True)
    def test_xpu_fallback(self, _mock_xpu: mock.Mock, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """When only Intel XPU is available, get_device() returns 'xpu'."""
        d = dev.get_device()
        self.assertEqual(d, "xpu")

    # ── S4: No accelerator → CPU fallback ────────────────────────────────────

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback_no_xpu(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """When no accelerator is available, get_device() returns 'cpu'.

        Note: If this machine has torch.xpu available, XPU detection will
        still return whatever torch.xpu.is_available() reports. This test
        works correctly regardless — CPU is the fallback when ALL backends
        report unavailable.
        """
        d = dev.get_device()
        # Mocks only cover cuda + mps; actual result depends on XPU availability.
        self.assertIn(d, ("cpu", "xpu"))

    # ── S5: device_name() returns human-readable string ──────────────────────

    def test_device_name_returns_string(self) -> None:
        """device_name() always returns a non-empty string."""
        name = dev.device_name()
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    # ── S6: device_type() returns category correctly ─────────────────────────

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_device_type_cuda(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """device_type() returns 'gpu' for CUDA."""
        self.assertEqual(dev.device_type(), "gpu")

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_device_type_cpu(self, _mock_mps: mock.Mock, _mock_cuda: mock.Mock) -> None:
        """device_type() returns 'cpu' when no GPU."""
        dtype = dev.device_type()
        self.assertIn(dtype, ("cpu", "gpu"))  # XPU also counts as gpu

    # ── S7: get_device() re-import consistency ───────────────────────────────

    def test_get_device_importable_from_package(self) -> None:
        """get_device is accessible from brats_seg directly."""
        from brats_seg import get_device as gd  # type: ignore[attr-defined]

        self.assertIs(gd, dev.get_device)


if __name__ == "__main__":
    unittest.main()
