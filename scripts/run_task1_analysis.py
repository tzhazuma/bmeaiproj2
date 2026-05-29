from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brats_seg.constants import DEFAULT_DATA_ROOT, MODALITIES
from brats_seg.data import discover_cases, limit_cases, preprocess_case, save_split_manifest, stable_split_cases, summarize_dataset
from brats_seg.preprocessing import canonicalize_segmentation
from brats_seg.visualization import save_multimodal_figure


def compute_contrast_statistics(image: np.ndarray, segmentation: np.ndarray) -> list[dict[str, float | str]]:
    seg = canonicalize_segmentation(segmentation)
    et_mask = seg == 4
    healthy_mask = seg == 0
    rows: list[dict[str, float | str]] = []
    for channel_index, modality in enumerate(MODALITIES):
        channel = image[channel_index]
        tumor_values = channel[et_mask]
        healthy_values = channel[healthy_mask]
        rows.append(
            {
                "modality": modality,
                "tumor_mean": float(tumor_values.mean()) if tumor_values.size else 0.0,
                "healthy_mean": float(healthy_values.mean()) if healthy_values.size else 0.0,
                "contrast_gap": float(tumor_values.mean() - healthy_values.mean()) if tumor_values.size and healthy_values.size else 0.0,
            }
        )
    return rows


def export_processed_slices(output_dir: Path, case_id: str, image: np.ndarray, regions: np.ndarray, max_slices: int = 16) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_indices = np.where(np.any(regions > 0, axis=(0, 2, 3)))[0]
    count = 0
    for slice_index in positive_indices[:max_slices]:
        np.savez_compressed(
            output_dir / f"{case_id}_slice_{slice_index:03d}.npz",
            image=image[:, slice_index],
            target=regions[:, slice_index],
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)  #change the place of dataset according to the local environment
    parser.add_argument("--output-dir", default="artifacts/task1")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.data_root)
    cases = limit_cases(cases, args.max_cases or None)
    splits = stable_split_cases(cases)
    save_split_manifest(splits, output_dir / "splits.json")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summarize_dataset(cases), indent=2), encoding="utf-8")

    contrast_rows: list[dict[str, float | str]] = []
    total_exported = 0
    for case in splits["train"][: args.num_examples]:
        processed = preprocess_case(case)
        image = processed["image"]
        seg = processed["seg"]
        regions = processed["regions"]
        save_multimodal_figure(image, regions, output_dir / "figures" / f"{case.case_id}.png")
        total_exported += export_processed_slices(output_dir / "processed_slices", case.case_id, image, regions)
        rows = compute_contrast_statistics(image, seg)
        for r in rows:
            r["case_id"] = case.case_id
        contrast_rows.extend(rows)

    with (output_dir / "contrast_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "modality", "tumor_mean", "healthy_mean", "contrast_gap"])
        writer.writeheader()
        writer.writerows(contrast_rows)

    modality_scores: dict[str, list[float]] = {modality: [] for modality in MODALITIES}
    for row in contrast_rows:
        modality_scores[str(row["modality"])].append(float(row["contrast_gap"]))

    # save a simple modality-level contrast plot (mean +/- std)
    means = [float(np.mean(modality_scores[m])) if modality_scores[m] else 0.0 for m in MODALITIES]
    stds = [float(np.std(modality_scores[m])) if modality_scores[m] else 0.0 for m in MODALITIES]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(MODALITIES))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax.bar(x, means, yerr=stds, capsize=6, color=colors[: len(MODALITIES)])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MODALITIES])
    ax.set_ylabel("Contrast gap (ET mean - healthy mean)")
    ax.set_title("Enhancing tumor contrast by modality")
    fig.tight_layout()
    fig.savefig(output_dir / "contrast_by_modality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    best_modality = max(modality_scores.items(), key=lambda item: np.mean(item[1]) if item[1] else float("-inf"))[0]
    (output_dir / "task1_summary.json").write_text(
        json.dumps({"best_et_contrast_modality": best_modality, "exported_slices": total_exported}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
