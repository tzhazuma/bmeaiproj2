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
python scripts/train_model.py --model unet --epochs 45 --output-dir artifacts/baseline --batch-size 128
python scripts/train_model.py --model attention_unet --epochs 10 --output-dir artifacts/attention --batch-size 32 --loss `dice_bce|dice_focal`
python scripts/train_model.py --model modality_attention_unet --epochs 10 --output-dir artifacts/modality_attention_unet  --batch-size 32
python scripts/train_all_models.py --model deep_attention_unet --epochs 10 --batch-size 32 --splits artifacts/preprocessed_cache/splits.json --cache-dir artifacts/preprocessed_cache --output-dir artifacts/deep_attention_unet
python scripts/visualize_predictions.py --model unet --checkpoint artifacts/baseline/best_model.pt --splits artifacts/preprocessed_cache/splits.json --output-dir artifacts/prediction_examples
python scripts/evaluate_model.py --model attention_unet --checkpoint artifacts/attention/best_model.pt --output-dir artifacts/eval_attention
```

`prepare_cache.py` samples raw slices before preprocessing when `--slices-per-case` is set, then writes preprocessed case data and `artifacts/preprocessed_cache/splits.json`; `train_model.py` reuses them and creates `--aug-samples-per-slice` random augmented samples per cached slice on the fly.

## SAM-like pretrained model fine-tuning

This project supports fine-tuning a SAM (Segment Anything Model) style pretrained model for BraTS segmentation via `scripts/finetune_sam.py`.

**Two modes:**

- **`--source vit`** (default) — pure-PyTorch ViT encoder, no external dependencies, random init. Use for testing the pipeline.
- **`--source sam`** — loads Meta's actual SAM weights. Requires the `segment-anything` package and a pretrained checkpoint.

```bash
# Standalone mode (random init, no download needed)
python scripts/finetune_sam.py --source vit --epochs 5

# Full SAM mode (requires checkpoint from Meta)
pip install brats-seg[sam]
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python scripts/finetune_sam.py --source sam \
    --checkpoint sam_vit_b_01ec64.pth --epochs 15
```

The SAM adapter (`src/brats_seg/models/sam_adapter.py`) automatically adapts SAM's 3-channel RGB image encoder to BraTS's 4-modality input by duplicating the nearest pretrained channel, and replaces SAM's mask decoder with a lightweight conv head outputting 3 regions (WT, TC, ET). The ViT encoder can be frozen (default) or fine-tuned end-to-end.

**Device auto-detection:** All training scripts use `get_device()` to automatically select CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon), XPU (Intel), or CPU fallback.

## Region-specific modality attention

`RegionModalityAttentionUNet2D` adds a modality-attention stage before Attention U-Net and can be trained with `python scripts/train_model.py --model modality_attention_unet`. For each output region `r` in `(WT, TC, ET)`, the model learns a separate softmax distribution over the four MRI modalities `(t1c, t1n, t2f, t2w)`:

```text
a_r = softmax(alpha_r),  a_r in R^4
x_r[c, h, w] = a_r[c] * x[c, h, w]
y_r = AttentionUNet_r(x_r)
y = concat(y_WT, y_TC, y_ET)
```

This lets different tumor regions emphasize different modalities before the image enters Attention U-Net. The learned weights are saved to `learned_modality_attention.json` after training.

## Deep Attention U-Net

`DeepSupAttentionUNet2D` is trained through `scripts/train_all_models.py`, because this script supports models with auxiliary deep-supervision outputs. The model returns the final prediction plus three auxiliary decoder predictions during training, and `train_all_models.py` combines their losses automatically.

Train it with the cached split:

```bash
python scripts/train_all_models.py --model deep_attention_unet --epochs 10 --batch-size 32 --splits artifacts/preprocessed_cache/splits.json --cache-dir artifacts/preprocessed_cache --output-dir artifacts/deep_attention_unet
```

The script writes `best_model.pt`, `best_metrics.json`, `history.json`, `history.csv`, and `loss_curve.png` under the chosen output directory.
