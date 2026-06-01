from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean_dice(metrics: dict) -> float:
    dice = metrics.get("dice", {})
    if not dice:
        return 0.0
    return sum(dice.values()) / len(dice)


def _mean_hd95(metrics: dict) -> float:
    hd95 = metrics.get("hd95", {})
    values = [v for v in hd95.values() if v >= 0]
    if not values:
        return -1.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="")
    parser.add_argument("--metrics-dir", default="")
    parser.add_argument("--metrics-manifest", default="")
    parser.add_argument("--metric", action="append", dest="metrics", default=[],
                        nargs=2, metavar=("MODEL", "PATH"),
                        help="Register a model/metrics file pair. Repeatable.")
    parser.add_argument("--baseline-metrics", default="")
    parser.add_argument("--attention-metrics", default="")
    parser.add_argument("--output-dir", default="artifacts/summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, dict] = {}

    # Collect from --metric MODEL PATH pairs
    for model_name, path in args.metrics:
        models[model_name] = json.loads(Path(path).read_text(encoding="utf-8"))

    # Backward-compatible: --baseline-metrics and --attention-metrics
    if args.baseline_metrics:
        models.setdefault("unet", json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8")))
    if args.attention_metrics:
        models.setdefault("attention_unet", json.loads(Path(args.attention_metrics).read_text(encoding="utf-8")))

    # Collect from --metrics-dir: {dir}/*/metrics.json or {dir}/metrics.json
    if args.metrics_dir:
        metrics_dir = Path(args.metrics_dir)
        if (metrics_dir / "metrics.json").exists():
            models.setdefault(metrics_dir.name or "model", json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8")))
        for subdir in sorted(metrics_dir.glob("*")):
            if subdir.is_dir() and (subdir / "metrics.json").exists():
                models.setdefault(subdir.name, json.loads((subdir / "metrics.json").read_text(encoding="utf-8")))

    # Collect from --metrics-manifest (JSON list of {"model": "...", "path": "..."})
    if args.metrics_manifest:
        manifest = json.loads(Path(args.metrics_manifest).read_text(encoding="utf-8"))
        for entry in manifest:
            if isinstance(entry, dict) and "model" in entry and "path" in entry:
                models.setdefault(entry["model"], json.loads(Path(entry["path"]).read_text(encoding="utf-8")))

    # Collect from --experiments-dir: scan for <model>/metrics.json
    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir)
        for subdir in sorted(exp_dir.glob("*")):
            if subdir.is_dir():
                metrics_file = subdir / "metrics.json"
                if metrics_file.exists():
                    models.setdefault(subdir.name, json.loads(metrics_file.read_text(encoding="utf-8")))

    # Rank by mean dice
    ranked = sorted(models.items(), key=lambda item: _mean_dice(item[1]), reverse=True)
    dice_ranks = {name: i + 1 for i, (name, _) in enumerate(ranked)}
    ranked_hd95 = sorted(
        [(n, m) for n, m in models.items() if _mean_hd95(m) >= 0],
        key=lambda item: _mean_hd95(item[1]),
    )
    hd95_ranks = {name: i + 1 for i, (name, _) in enumerate(ranked_hd95)}

    # Build per-region CSV rows
    region_names = ["WT", "TC", "ET"]
    csv_rows: list[dict] = []

    for model_name, metrics in ranked:
        dice = metrics.get("dice", {})
        hd95 = metrics.get("hd95", {})
        row: dict = {
            "model": model_name,
            "loss": metrics.get("loss", 0.0),
            "mean_dice": round(_mean_dice(metrics), 5),
            "dice_rank": dice_ranks.get(model_name, -1),
            "mean_hd95": round(_mean_hd95(metrics), 5),
            "hd95_rank": hd95_ranks.get(model_name, -1),
        }
        for region in region_names:
            row[f"dice_{region}"] = round(dice.get(region, 0.0), 5)
        for region in region_names:
            row[f"hd95_{region}"] = round(hd95.get(region, -1.0), 5)
        csv_rows.append(row)

    # Write ranked CSV
    fieldnames = list(csv_rows[0].keys()) if csv_rows else ["model"]
    with (output_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write summary.json
    summary = {
        "rankings": {
            "by_mean_dice": [name for name, _ in ranked],
            "by_mean_hd95": [name for name, _ in ranked_hd95],
        },
        "models": {name: metrics for name, metrics in models.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write backward-compatible per-region comparison if exactly 2 models
    if len(models) == 2:
        (m1, v1), (m2, v2) = list(models.items())
        legacy_rows = []
        for region in region_names:
            legacy_rows.append({
                "region": region,
                f"{m1}_dice": round(v1["dice"].get(region, 0.0), 5),
                f"{m2}_dice": round(v2["dice"].get(region, 0.0), 5),
                "dice_delta": round(v2["dice"].get(region, 0.0) - v1["dice"].get(region, 0.0), 5),
                f"{m1}_hd95": round(v1["hd95"].get(region, -1.0), 5),
                f"{m2}_hd95": round(v2["hd95"].get(region, -1.0), 5),
            })
        with (output_dir / "experiment_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(legacy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(legacy_rows)


if __name__ == "__main__":
    main()
