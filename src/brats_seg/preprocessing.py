from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    def as_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_min, self.z_max),
            slice(self.y_min, self.y_max),
            slice(self.x_min, self.x_max),
        )


def zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)
    masked = volume[mask]
    mean = float(masked.mean())
    std = float(masked.std())
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / max(std, eps)
    return normalized


def canonicalize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    seg = np.rint(segmentation).astype(np.int16, copy=False)
    unique_values = set(np.unique(seg).tolist())
    if 4 in unique_values:
        return seg
    if 3 in unique_values and 4 not in unique_values:
        seg = seg.copy()
        seg[seg == 3] = 4
    return seg


def segmentation_to_regions(segmentation: np.ndarray) -> np.ndarray:  # convert the segmentation map to three binary masks for the three regions of interest (WT, TC, ET)
    seg = canonicalize_segmentation(segmentation)
    wt = seg > 0
    tc = np.isin(seg, (1, 4))
    et = seg == 4
    return np.stack((wt, tc, et), axis=0).astype(np.float32)


def compute_foreground_bbox(volumes: np.ndarray, margin: int = 0) -> BoundingBox:
    if volumes.ndim != 4:
        raise ValueError(f"Expected volumes with shape (C, Z, Y, X), got {volumes.shape}")
    mask = np.any(volumes != 0, axis=0)
    if not np.any(mask):
        z, y, x = volumes.shape[1:]
        return BoundingBox(0, z, 0, y, 0, x)
    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    z, y, x = volumes.shape[1:]
    return BoundingBox(
        max(0, int(z_min) - margin),
        min(z, int(z_max) + margin),
        max(0, int(y_min) - margin),
        min(y, int(y_max) + margin),
        max(0, int(x_min) - margin),
        min(x, int(x_max) + margin),
    )


def crop_volumes(volumes: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return volumes[(slice(None),) + bbox.as_slices()]


def crop_mask(mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return mask[bbox.as_slices()]


def pad_or_crop_2d(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    h, w = array.shape[-2:]
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    cropped = array[..., start_h : start_h + min(h, target_h), start_w : start_w + min(w, target_w)]
    out_shape = array.shape[:-2] + (target_h, target_w)
    output = np.zeros(out_shape, dtype=array.dtype)
    out_h = cropped.shape[-2]
    out_w = cropped.shape[-1]
    pad_h = (target_h - out_h) // 2
    pad_w = (target_w - out_w) // 2
    output[..., pad_h : pad_h + out_h, pad_w : pad_w + out_w] = cropped
    return output
