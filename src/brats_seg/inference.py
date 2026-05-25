from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .metrics import dice_per_region, hd95_per_region, threshold_predictions
from .preprocessing import pad_or_crop_2d


@dataclass
class CasePredictionResult:
    case_id: str
    image: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    dice: dict[str, float]
    hd95: dict[str, float]


@torch.no_grad()
def predict_case_regions(
    model: torch.nn.Module,
    image_volume: np.ndarray,
    target_volume: np.ndarray,
    device: str,
    case_id: str = "",
    target_shape: tuple[int, int] = (160, 160),
    threshold: float = 0.5,
) -> CasePredictionResult:
    model.eval()
    num_slices = image_volume.shape[1]
    aligned_image = np.zeros((image_volume.shape[0], num_slices, *target_shape), dtype=np.float32)
    aligned_target = np.zeros((target_volume.shape[0], num_slices, *target_shape), dtype=np.uint8)
    prediction = np.zeros((target_volume.shape[0], num_slices, *target_shape), dtype=np.uint8)
    for slice_index in range(num_slices):
        image = pad_or_crop_2d(image_volume[:, slice_index], target_shape)
        target = pad_or_crop_2d(target_volume[:, slice_index], target_shape)
        aligned_image[:, slice_index] = image
        aligned_target[:, slice_index] = target.astype(np.uint8)
        tensor = torch.from_numpy(image).float().unsqueeze(0).to(device=device)
        logits = model(tensor)
        pred = threshold_predictions(logits, threshold=threshold)[0]
        prediction[:, slice_index] = pred
    return CasePredictionResult(
        case_id=case_id,
        image=aligned_image,
        prediction=prediction,
        target=aligned_target,
        dice=dice_per_region(prediction, aligned_target),
        hd95=hd95_per_region(prediction, aligned_target),
    )
