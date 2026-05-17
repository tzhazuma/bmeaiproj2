from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--attention-metrics", required=True)
    parser.add_argument("--output-dir", default="artifacts/summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    attention = json.loads(Path(args.attention_metrics).read_text(encoding="utf-8"))

    rows = []
    for region in ["WT", "TC", "ET"]:
        rows.append(
            {
                "region": region,
                "baseline_dice": baseline["dice"][region],
                "attention_dice": attention["dice"][region],
                "dice_delta": attention["dice"][region] - baseline["dice"][region],
                "baseline_hd95": baseline["hd95"][region],
                "attention_hd95": attention["hd95"][region],
            }
        )

    with (output_dir / "experiment_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "better_model_by_mean_dice": "attention_unet"
        if sum(attention["dice"].values()) > sum(baseline["dice"].values())
        else "unet",
        "baseline": baseline,
        "attention": attention,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
