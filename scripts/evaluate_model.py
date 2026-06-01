from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import BraTSSliceDataset, discover_cases, limit_cases, load_split_manifest, preprocess_case, save_split_manifest, stable_split_cases
from brats_seg.models import (
    UNet2D,
    AttentionUNet2D,
    DeepSupAttentionUNet2D,
    MedNeXt2D,
    ResEncUNet2D,
    SegResNet2D,
    SwinUNETR2D,
    RegionModalityAttentionUNet2D,
)
from brats_seg.training import create_device, evaluate_model
from brats_seg.visualization import choose_representative_slice
from brats_seg.visualization import save_prediction_figure

MODEL_CHOICES = (
    "unet",
    "attention_unet",
    "deep_sup_attention_unet",
    "mednext",
    "resenc_unet",
    "segresnet",
    "swin_unetr",
    "region_modality_attention_unet",
)


def create_model(name: str) -> torch.nn.Module:
    """Create a model instance by name (use_checkpoint=False for eval)."""
    if name == "unet":
        return UNet2D()
    elif name == "attention_unet":
        return AttentionUNet2D()
    elif name == "deep_sup_attention_unet":
        return DeepSupAttentionUNet2D()
    elif name == "mednext":
        return MedNeXt2D(use_checkpoint=False)
    elif name == "resenc_unet":
        return ResEncUNet2D(use_checkpoint=False)
    elif name == "segresnet":
        return SegResNet2D()
    elif name == "swin_unetr":
        return SwinUNETR2D(use_checkpoint=False)
    elif name == "region_modality_attention_unet":
        return RegionModalityAttentionUNet2D()
    else:
        raise ValueError(f"Unknown model: {name}. Choices: {MODEL_CHOICES}")


def load_model_state(model: torch.nn.Module, checkpoint: dict[str, Any] | str, device: str = "cpu") -> None:
    """Load model state from checkpoint dict, handling multiple key conventions.

    Supports checkpoints with keys: ``backbone``, ``model_state_dict``, ``state_dict``,
    ``model``, or a plain state_dict (no known wrapper key).
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(str(checkpoint), map_location=device, weights_only=False)

    state_dict: dict[str, Any] | None = None

    if isinstance(checkpoint, dict):
        # Try known wrapper keys in priority order
        for key in ("backbone", "model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        # If no known wrapper key found, but dict contains param-like keys,
        # treat the whole dict as a state_dict
        if state_dict is None:
            first_key = next(iter(checkpoint), "")
            if "." in first_key or any(k.startswith("stem.") or k.startswith("enc") for k in checkpoint):
                state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()

    if state_dict is None:
        raise ValueError("Could not extract state_dict from checkpoint")

    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/eval")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="unet")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    else:
        cases = discover_cases(args.data_root)
        cases = limit_cases(cases, args.max_cases or None)
        splits = stable_split_cases(cases)
        save_split_manifest(splits, output_dir / "splits.json")

    test_dataset = BraTSSliceDataset(splits["test"], include_empty=False, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    model = create_model(args.model)
    device = create_device()
    load_model_state(model, args.checkpoint, device)
    model.to(device)
    summary = evaluate_model(model, test_loader, device)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    from brats_seg.inference import predict_case_regions

    sample_case = splits["test"][0]
    processed = preprocess_case(sample_case)
    result = predict_case_regions(model, processed["image"], processed["regions"], device, case_id=sample_case.case_id)
    slice_index = choose_representative_slice(processed["regions"])
    save_prediction_figure(processed["image"], processed["regions"], result.prediction, output_dir / f"{sample_case.case_id}_prediction.png", slice_index)


if __name__ == "__main__":
    main()
