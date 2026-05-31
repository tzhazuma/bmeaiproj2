# BraTS 2023 Glioma Segmentation Project

This repository implements the coursework pipeline for BraTS-based multi-modal MRI glioma segmentation using the local dataset at `/mnt/d/brats2023` and the  Python environment with torch.

## Implemented scope

- Task 1: NIfTI loading, z-score normalization, foreground cropping, WT/TC/ET label transformation, visualization, and ET contrast analysis.
- Task 2: 2D slice dataset, augmentation, baseline U-Net, Dice+BCE loss, training, validation Dice, and loss-curve export.
- Task 3: Attention U-Net upgrade, HD95 evaluation, and prediction figure generation.

## Quick start

```bash
python -m pip install -e .
python scripts/run_task1_analysis.py
python scripts/prepare_cache.py --cache-dir artifacts/preprocessed_cache --slices-per-case 32 --overwrite
python scripts/train_model.py --model unet --epochs 3 --output-dir artifacts/baseline --aug-samples-per-slice 2
python scripts/train_model.py --model attention_unet --epochs 3 --output-dir artifacts/attention
python scripts/visualize_predictions.py --model unet --checkpoint artifacts/baseline/best_model.pt --splits artifacts/preprocessed_cache/splits.json --output-dir artifacts/prediction_examples
python scripts/evaluate_model.py --model attention_unet --checkpoint artifacts/attention/best_model.pt --output-dir artifacts/eval_attention
```

`prepare_cache.py` samples raw slices before preprocessing when `--slices-per-case` is set, then writes preprocessed case data and `artifacts/preprocessed_cache/splits.json`; `train_model.py` reuses them and creates `--aug-samples-per-slice` random augmented samples per cached slice on the fly.
