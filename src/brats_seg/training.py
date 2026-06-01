from __future__ import annotations

import csv
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import dice_bce_loss, dice_focal_loss
from .metrics import dice_per_region, hd95_per_region, threshold_predictions


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_workers: int = 0
    threshold: float = 0.5
    loss: str = "dice_bce"


class EvalSummary(TypedDict):
    loss: float
    dice: dict[str, float]
    hd95: dict[str, float]


def create_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_loss_function(loss_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_name == "dice_bce":
        return dice_bce_loss
    if loss_name == "dice_focal":
        return dice_focal_loss
    raise ValueError(f"Unknown loss function: {loss_name}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    loss_name: str = "dice_bce",
) -> float:
    train = optimizer is not None
    model.train(train)
    loss_fn = get_loss_function(loss_name)
    losses: list[float] = []
    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        image = batch["image"].float().to(device=device)
        target = batch["target"].float().to(device=device)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss = loss_fn(logits, target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        iterator.set_postfix(loss=f"{losses[-1]:.4f}")
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    loss_name: str = "dice_bce",
) -> EvalSummary:
    model.eval()
    loss_fn = get_loss_function(loss_name)
    losses: list[float] = []
    dice_scores: list[dict[str, float]] = []
    hd95_scores: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].float().to(device=device)
        target = batch["target"].float().to(device=device)
        logits = model(image)
        loss = loss_fn(logits, target)
        losses.append(float(loss.item()))
        preds = threshold_predictions(logits, threshold=threshold)
        truth = target.detach().cpu().numpy().astype(np.uint8)
        for pred_item, truth_item in zip(preds, truth, strict=True):
            dice_scores.append(dice_per_region(pred_item, truth_item))
            hd95_scores.append(hd95_per_region(pred_item, truth_item))
    summary: EvalSummary = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dice": _average_metric_dicts(dice_scores),
        "hd95": _average_metric_dicts(hd95_scores),
    }
    return summary


def _average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([item[key] for item in metric_dicts])) for key in keys}


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    config: TrainConfig,
) -> dict[str, list[float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = create_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_metrics: EvalSummary = {"loss": 0.0, "dice": {}, "hd95": {}}
    logger.info(
        "Starting training: device=%s, epochs=%d, batch_size=%d, lr=%.3g, loss=%s",
        device,
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.loss,
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, loss_name=config.loss)
        val_summary = evaluate_model(model, val_loader, device, threshold=config.threshold, loss_name=config.loss)
        val_loss = val_summary["loss"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        _append_history_row(output_dir / "history.csv", epoch, train_loss, val_loss, val_summary)
        dice = val_summary["dice"]
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | dice(WT=%.4f, TC=%.4f, ET=%.4f)",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            dice.get("WT", 0.0),
            dice.get("TC", 0.0),
            dice.get("ET", 0.0),
        )
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = val_summary
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("New best checkpoint saved: val_loss=%.4f", best_val)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    logger.info("Training finished. Best val_loss=%.4f", best_val)
    return history


def _append_history_row(path: Path, epoch: int, train_loss: float, val_loss: float, val_summary: EvalSummary) -> None:
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "dice_WT", "dice_TC", "dice_ET", "hd95_WT", "hd95_TC", "hd95_ET"])
        dice = val_summary["dice"]
        hd95 = val_summary["hd95"]
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            dice.get("WT", 0.0),
            dice.get("TC", 0.0),
            dice.get("ET", 0.0),
            hd95.get("WT", 0.0),
            hd95.get("TC", 0.0),
            hd95.get("ET", 0.0),
        ])
