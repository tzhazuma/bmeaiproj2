from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = logits.sigmoid()
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.7) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return dice_weight * dice + (1.0 - dice_weight) * bce


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = logits.sigmoid()
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = alpha * (1.0 - pt).pow(gamma)
    return (focal_weight * bce).mean()


def dice_focal_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    dice = dice_loss(logits, targets)
    focal = focal_loss(logits, targets)
    return dice_weight * dice + (1.0 - dice_weight) * focal
