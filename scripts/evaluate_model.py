from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import BraTSSliceDataset, discover_cases, limit_cases, load_split_manifest, preprocess_case, save_split_manifest, stable_split_cases
from brats_seg.metrics import threshold_predictions
from brats_seg.models import AttentionUNet2D, UNet2D
from brats_seg.training import create_device, evaluate_model
from brats_seg.visualization import choose_representative_slice
from brats_seg.visualization import save_prediction_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/eval")
    parser.add_argument("--model", choices=("unet", "attention_unet"), default="unet")
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
    model = UNet2D() if args.model == "unet" else AttentionUNet2D()
    device = create_device()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
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
