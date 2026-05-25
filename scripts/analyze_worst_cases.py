from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import discover_cases, limit_cases, load_split_manifest, preprocess_case, save_split_manifest, stable_split_cases
from brats_seg.inference import predict_case_regions
from brats_seg.models import AttentionUNet2D, UNet2D
from brats_seg.training import create_device
from brats_seg.visualization import choose_representative_slice, save_prediction_figure


class FailureHeuristics(TypedDict):
    t1c_tumor_mean: float
    t1c_healthy_mean: float
    t1c_contrast_gap: float
    lesion_fraction: float
    foreground_std: float


class CaseMetricRow(FailureHeuristics):
    case_id: str
    mean_dice: float
    dice_WT: float
    dice_TC: float
    dice_ET: float
    hd95_WT: float
    hd95_TC: float
    hd95_ET: float
    suspected_reason: str


def compute_failure_reason(image: np.ndarray, seg: np.ndarray) -> tuple[str, FailureHeuristics]:
    et_mask = seg == 4
    healthy_mask = seg == 0
    t1c = image[0]
    tumor_mean = float(t1c[et_mask].mean()) if np.any(et_mask) else 0.0
    healthy_mean = float(t1c[healthy_mask].mean()) if np.any(healthy_mask) else 0.0
    contrast_gap = abs(tumor_mean - healthy_mean)
    lesion_fraction = float(np.mean(seg > 0))
    foreground_std = float(t1c[t1c != 0].std()) if np.any(t1c != 0) else 0.0
    if lesion_fraction < 0.01:
        reason = "small lesion size"
    elif contrast_gap < 0.35:
        reason = "low contrast"
    elif foreground_std > 1.3:
        reason = "possible artifact or intensity instability"
    else:
        reason = "mixed factors"
    return reason, {
        "t1c_tumor_mean": tumor_mean,
        "t1c_healthy_mean": healthy_mean,
        "t1c_contrast_gap": contrast_gap,
        "lesion_fraction": lesion_fraction,
        "foreground_std": foreground_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/worst_cases")
    parser.add_argument("--model", choices=("unet", "attention_unet"), default="unet")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    else:
        cases = limit_cases(discover_cases(args.data_root), args.max_cases or None)
        splits = stable_split_cases(cases)
        save_split_manifest(splits, output_dir / "splits.json")

    model = UNet2D() if args.model == "unet" else AttentionUNet2D()
    device = create_device()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    model.to(device)

    rows: list[CaseMetricRow] = []
    results_cache: dict[str, object] = {}
    for case in splits["test"]:
        processed = preprocess_case(case)
        result = predict_case_regions(model, processed["image"], processed["regions"], device, case_id=case.case_id)
        results_cache[case.case_id] = result
        mean_dice = float(np.mean(list(result.dice.values())))
        reason, heuristics = compute_failure_reason(processed["image"], processed["seg"])
        row: CaseMetricRow = {
            "case_id": case.case_id,
            "mean_dice": mean_dice,
            "dice_WT": result.dice["WT"],
            "dice_TC": result.dice["TC"],
            "dice_ET": result.dice["ET"],
            "hd95_WT": result.hd95["WT"],
            "hd95_TC": result.hd95["TC"],
            "hd95_ET": result.hd95["ET"],
            "suspected_reason": reason,
            **heuristics,
        }
        rows.append(row)

    rows.sort(key=lambda item: item["mean_dice"])
    with (output_dir / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    worst_cases = rows[: args.top_k]
    (output_dir / "worst_cases.json").write_text(json.dumps(worst_cases, indent=2), encoding="utf-8")

    for item in worst_cases:
        case_id = str(item["case_id"])
        result = results_cache[case_id]
        slice_index = choose_representative_slice(result.target)
        save_prediction_figure(
            result.image,
            result.target,
            result.prediction,
            output_dir / f"{case_id}_worst_case.png",
            slice_index,
        )


if __name__ == "__main__":
    main()
