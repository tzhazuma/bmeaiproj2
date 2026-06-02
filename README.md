# BraTS 2023 Glioma Segmentation Project

This repository implements the coursework pipeline for BraTS-based multi-modal MRI glioma segmentation. The dataset is downloaded from HuggingFace (`Angelou0516/brats2023-gli-dataset`) and trained on 4× NVIDIA RTX 5090 GPUs (32GB each) with bf16 mixed precision.

## Implemented scope

- Task 1: NIfTI loading, z-score normalization, foreground cropping, WT/TC/ET label transformation, visualization, and ET contrast analysis.
- Task 2: 2D slice dataset, augmentation, baseline U-Net, Dice+BCE loss, training, validation Dice, and loss-curve export.
- Task 3: Attention U-Net upgrade, HD95 evaluation, and prediction figure generation.
- **Task 4**: Pretrained model fine-tuning (SAM, SegFormer+Lora) with bf16 AMP and multi-GPU parallel training.

## Quick start

```bash
# 1. Install the package
python -m pip install -e .

# 2. Download 30-case BraTS subset
python scripts/download_brats_subset.py

# 3. Prepare preprocessed cache
python scripts/prepare_cache.py --data-root data/brats2023 --cache-dir artifacts/preprocessed_cache --slices-per-case 32 --overwrite

# 4. Train models (example commands)
python scripts/train_model.py --model unet --epochs 45 --output-dir artifacts/baseline --batch-size 128 --data-root data/brats2023
python scripts/train_model.py --model attention_unet --epochs 10 --output-dir artifacts/attention --batch-size 32 --data-root data/brats2023
python scripts/train_all_models.py --model swin_unetr --epochs 15 --batch-size 64 --amp --data-root data/brats2023 --splits artifacts/preprocessed_cache/splits.json --cache-dir artifacts/preprocessed_cache --output-dir artifacts/swin_unetr

# 5. Evaluate
python scripts/evaluate_model.py --model attention_unet --checkpoint artifacts/attention/best_model.pt --output-dir artifacts/eval_attention --data-root data/brats2023
```

**Note**: All commands now accept `--data-root` to specify the BraTS dataset location. Default: `data/brats2023`.

## SAM-like pretrained model fine-tuning

This project supports fine-tuning a SAM (Segment Anything Model) style pretrained model for BraTS segmentation via `scripts/finetune_sam.py`.

**Key features (v2):**
- `--amp` (default on): bf16 mixed precision via `torch.amp.autocast`
- `--grad-accum-steps N`: Gradient accumulation for larger effective batch sizes
- `--gpu 0`: Per-GPU device selection via `CUDA_VISIBLE_DEVICES`
- `--resume checkpoint.pt`: Resume training from saved checkpoint
- `--batch-size 64`: Larger default batch size for 32GB GPUs
- GPU memory reporting in progress bar (alloc/cache)

**Two modes:**

- **`--source vit`** (default) — pure-PyTorch ViT encoder, no external dependencies, random init.
- **`--source sam`** — loads Meta's actual SAM weights. Requires `segment-anything` and a pretrained checkpoint.

```bash
# Standalone mode (ViT-B, bf16, bs=64)
python scripts/finetune_sam.py --source vit --epochs 15 --batch-size 64 --amp

# Full SAM mode (requires checkpoint from Meta)
pip install segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python scripts/finetune_sam.py --source sam \
    --checkpoint sam_vit_b_01ec64.pth --epochs 15 --batch-size 32 --amp
```

## SegFormer LoRA Fine-tuning (PEFT)

Fine-tune HuggingFace SegFormer models with Low-Rank Adaptation (LoRA) for parameter-efficient transfer learning on BraTS.

```bash
python scripts/finetune_segformer.py \
    --model nvidia/mit-b0 \
    --data-root data/brats2023 \
    --output-dir artifacts/segformer_b0_lora \
    --epochs 20 --batch-size 32 --lr 5e-4 --amp
```

**Key features:**
- PEFT LoRA (r=16, α=32) targeting `query` and `value` attention projections
- `--use-4channel`: Widen first conv layer to accept 4 BraTS modalities natively
- `--grad-accum-steps`: Gradient accumulation for large effective batch
- `--gpu`: Per-GPU device selection
- `--resume`: Resume from saved LoRA adapter checkpoint

### Multi-GPU Parallel Training

Run 4 models simultaneously on 4 GPUs for maximum utilization:

```bash
bash scripts/multigpu_launch.sh
```

This launches:
| GPU | Model | Batch | Config |
|-----|-------|-------|--------|
| 0 | SAM ViT-B (full FT) | 64 | bf16, 88.3M trainable |
| 1 | SAM ViT-B (frozen enc) | 96 | bf16, 2.4M trainable |
| 2 | SwinUNETR2D | 64 | bf16, AMP |
| 3 | SegFormer-B0 LoRA | 32 | bf16, PEFT |

## Architecture Comparison (7 models + pretrained)

| Model | Params | Dice WT | Dice TC | Dice ET |
|-------|--------|---------|---------|---------|
| UNet2D | 7.76M | 0.871 | 0.726 | 0.723 |
| AttentionUNet2D | 7.85M | 0.886 | 0.738 | 0.806 |
| ResEncUNet2D | 7.66M | 0.886 | 0.689 | 0.779 |
| SegResNet2D | 4.15M | **0.904** | 0.665 | 0.733 |
| SwinUNETR2D | 6.90M | 0.898 | 0.586 | 0.729 |
| DeepSupAttnUNet2D | 8.63M | 0.865 | **0.784** | **0.814** |
| --- | --- | --- | --- | --- |
| **Pretrained/Fine-tuned** | | | | |
| **SegFormer-B0 LoRA** | 4.2M | 0.678 | **0.687** | **0.689** |
| SwinUNETR2D (scratch) | 6.9M | 0.684 | 0.540 | 0.678 |
| MedNeXt2D Light | 1.1M | 0.594 | 0.622 | 0.645 |
| ResEncUNet2D | 7.7M | 0.636 | 0.577 | 0.597 |
| SAM ViT-B (full FT) | 88.3M | 0.575 | 0.555 | 0.540 |
| SAM ViT-B (frozen) | 88.3M | 0.498 | 0.394 | 0.377 |

**Best model**: SegFormer-B0 with LoRA (461K trainable params, mean Dice 0.685). All models trained 10-15 epochs on 30 BraTS cases, bf16 AMP, zero crashes.

## Environment

- Python 3.14, PyTorch 2.11+cu128, Transformers 5.9.0
- PEFT 0.19.1, Accelerate 1.13.0
- 4× NVIDIA RTX 5090 (32GB VRAM each)
- CUDA 12.9, Driver 575.64.03
