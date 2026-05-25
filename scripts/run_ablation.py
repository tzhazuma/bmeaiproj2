#!/usr/bin/env python3
"""Run all ablation experiments and collect results."""
import subprocess, json, sys
from pathlib import Path

ROOT = Path("/home/azuma/bmeaiproj2")
VENV = ROOT / ".venv/bin/python"
DATA = ROOT / "data/brats2023"
ARTIFACTS = ROOT / "artifacts/ablation"
SCRIPT = ROOT / "scripts/train_all_models.py"

BASE_ARGS = f"--data-root {DATA} --max-cases 8 --epochs 8 --amp"

MODELS = ["unet", "attention_unet", "resenc_unet", "segresnet", "mednext", "swin_unetr", "deep_attention_unet"]

results = {}

for model in MODELS:
    out = ARTIFACTS / model
    cmd = f"{VENV} {SCRIPT} --model {model} {BASE_ARGS} --output-dir {out}"
    print(f"\n{'='*60}\nTraining {model}...\n{'='*60}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr[-500:]}")
        continue
    
    metrics_file = out / "best_metrics.json"
    history_file = out / "history.csv"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        results[model] = metrics
        print(f"  => {model}: Dice WT={metrics['dice'].get('WT',0):.4f} TC={metrics['dice'].get('TC',0):.4f} ET={metrics['dice'].get('ET',0):.4f}")

summary_file = ARTIFACTS / "all_results.json"
summary_file.parent.mkdir(parents=True, exist_ok=True)
summary_file.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {summary_file}")
