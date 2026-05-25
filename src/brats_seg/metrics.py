from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from typing import cast

from .constants import REGION_NAMES


def dice_per_region(prediction: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, name in enumerate(REGION_NAMES):
        pred = prediction[index].astype(bool)
        truth = target[index].astype(bool)
        intersection = float(np.logical_and(pred, truth).sum())
        denom = float(pred.sum() + truth.sum())
        scores[name] = (2.0 * intersection + smooth) / (denom + smooth)
    return scores


def threshold_predictions(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (logits.sigmoid().detach().cpu().numpy() >= threshold).astype(np.uint8)


def _surface_distances(result: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not np.any(result) and not np.any(reference):
        return np.zeros(1, dtype=np.float32)
    if not np.any(result) or not np.any(reference):
        return np.array([np.inf], dtype=np.float32)
    result_surface = np.logical_xor(result, binary_erosion(result))
    reference_surface = np.logical_xor(reference, binary_erosion(reference))
    dt = cast(np.ndarray, distance_transform_edt(~reference_surface))
    return np.asarray(dt[result_surface], dtype=np.float32)


def hd95_per_region(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, name in enumerate(REGION_NAMES):
        pred = prediction[index].astype(bool)
        truth = target[index].astype(bool)
        if not np.any(pred) and not np.any(truth):
            scores[name] = 0.0
            continue
        if not np.any(pred) or not np.any(truth):
            scores[name] = -1.0
            continue
        d1 = _surface_distances(pred, truth)
        d2 = _surface_distances(truth, pred)
        joined = np.concatenate((d1, d2))
        if np.isinf(joined).any():
            scores[name] = -1.0
            continue
        scores[name] = float(np.percentile(joined, 95))
    return scores
