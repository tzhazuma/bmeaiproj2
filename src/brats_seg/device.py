"""Device auto-detection for PyTorch across CUDA, ROCm, MPS, Intel XPU, and CPU.

PyTorch ROCm builds expose AMD GPUs through the CUDA API layer, so
``torch.device()`` accepts ``"cuda"`` on both NVIDIA and AMD systems.
The ``_rocm_available()`` helper distinguishes the two at the human-readable
``device_name()`` level.

Usage::

    from brats_seg.device import get_device, device_name, device_type

    dev = get_device()          # "cuda" | "mps" | "xpu" | "cpu"
    typ = device_type()         # "gpu" | "cpu"
    name = device_name()        # human-readable string
"""

from __future__ import annotations

import torch


def _rocm_available() -> bool:
    """Check if PyTorch is built with ROCm (AMD GPU) support."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_device() -> str:
    """Detect the best available torch device.

    On ROCm (AMD), PyTorch uses ``"cuda"`` as the device string —
    same as for NVIDIA CUDA — because ROCm provides a CUDA API
    compatibility layer.

    Priority order (first available wins):
    1. CUDA / ROCm (NVIDIA GPU / AMD GPU via ROCm)
    2. MPS  (Apple Silicon / Metal Performance Shaders)
    3. XPU  (Intel GPU via ``intel-extension-for-pytorch``)
    4. CPU  (universal fallback)

    Returns:
        Device string suitable for ``torch.device()``:
        ``"cuda"``, ``"mps"``, ``"xpu"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    # Intel XPU — requires ``import intel_extension_for_pytorch``
    # but the xpu backend registers itself as ``torch.xpu``.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    return "cpu"


def device_type() -> str:
    """Return ``"gpu"`` if any GPU-like accelerator is active, else ``"cpu"``."""
    dev = get_device()
    gpu_devices = ("cuda", "mps", "xpu")
    return "gpu" if dev in gpu_devices else "cpu"


def device_name() -> str:
    """Return a human-readable device name string.

    Examples::

        "AMD Radeon RX 7900 XTX"    (ROCm)
        "NVIDIA GeForce RTX 4090"   (CUDA)
        "Apple M2 Max"              (MPS)
        "Intel Data Center GPU"     (XPU)
        "CPU"                       (fallback)
    """
    dev = get_device()

    if dev == "cuda":
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else "GPU"
        if _rocm_available():
            return f"{name} (ROCm)"
        return name

    if dev == "mps":
        return "Apple Silicon (MPS)"

    if dev == "xpu":
        count = torch.xpu.device_count()
        name = torch.xpu.get_device_name(0) if count > 0 else "Intel GPU"
        return name

    return "CPU"
