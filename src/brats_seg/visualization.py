from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import MODALITIES, REGION_NAMES


def choose_representative_slice(regions: np.ndarray) -> int:
    per_slice = regions[0].sum(axis=(1, 2))
    return int(np.argmax(per_slice))


def save_multimodal_figure(
    image: np.ndarray,
    segmentation: np.ndarray,
    output_path: str | Path,
    slice_index: int | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if slice_index is None:
        slice_index = choose_representative_slice(segmentation)
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(18, 4))
    overlay = np.argmax(segmentation[:, slice_index], axis=0)
    for idx, modality in enumerate(MODALITIES):
        axes[idx].imshow(image[idx, slice_index], cmap="gray")
        axes[idx].imshow(np.ma.masked_where(segmentation[0, slice_index] == 0, overlay), alpha=0.35, cmap="autumn")
        axes[idx].set_title(modality.upper())
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prediction_figure(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    slice_index: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    axes[0, 0].imshow(image[0, slice_index], cmap="gray")
    axes[0, 0].set_title("T1C")
    axes[0, 0].axis("off")
    for row, (name, truth, pred) in enumerate(zip(REGION_NAMES, target[:, slice_index], prediction[:, slice_index], strict=True)):
        axes[row, 1].imshow(truth, cmap="viridis")
        axes[row, 1].set_title(f"GT {name}")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pred, cmap="magma")
        axes[row, 2].set_title(f"Pred {name}")
        axes[row, 2].axis("off")
    axes[1, 0].imshow(image[2, slice_index], cmap="gray")
    axes[1, 0].set_title("T2F")
    axes[1, 0].axis("off")
    axes[2, 0].imshow(image[3, slice_index], cmap="gray")
    axes[2, 0].set_title("T2W")
    axes[2, 0].axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_loss_curve(history: dict[str, list[float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(epochs), history["train_loss"], label="train")
    ax.plot(list(epochs), history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
