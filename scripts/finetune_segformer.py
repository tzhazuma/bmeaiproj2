#!/usr/bin/env python3
"""Fine-tune HuggingFace SegFormer models on BraTS brain-tumor segmentation with PEFT/LoRA.

SegFormer expects 3-channel RGB input.  BraTS provides 4 modalities (t1c, t1n, t2f,
t2w).  By default we drop the least discriminating t1n channel and feed
{t1c, t2f, t2w} as pseudo-RGB.  Use ``--use-4channel`` to surgically widen the
first patch-embedding Conv2d to accept all four modalities instead.

Usage::

    ~/venv314/bin/python scripts/finetune_segformer.py \\
        --model nvidia/mit-b0 \\
        --data-root /home/tangzh/bmeaiproj2/data/brats2023 \\
        --output-dir artifacts/segformer_b0_lora \\
        --epochs 20 --batch-size 32 --lr 5e-4

References
----------
* Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation
  with Transformers" (NeurIPS 2021).
* HuggingFace ``transformers.SegformerForSemanticSegmentation``
* PEFT LoRA: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"
  (ICLR 2022).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from accelerate import Accelerator

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from peft import LoraConfig, get_peft_model

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import (
    BraTSSliceDataset,
    discover_cases,
    limit_cases,
    stable_split_cases,
)
from brats_seg.device import device_name, get_device
from brats_seg.losses import dice_bce_loss
from brats_seg.metrics import dice_per_region, hd95_per_region, threshold_predictions
from brats_seg.visualization import save_loss_curve

# ── constants ────────────────────────────────────────────────────────────

# BraTS modality order:  t1c, t1n, t2f, t2w
# Drop t1n (idx 1) -- keep the three most informative channels.
_DEFAULT_CHANNELS = (0, 2, 3)

_MODEL_CHOICES = tuple(f"nvidia/mit-b{i}" for i in range(6))

_MIXED_PRECISION_CHOICES = ("no", "fp16", "bf16")


# ── dataset wrapper ──────────────────────────────────────────────────────

class _SegFormerDatasetWrapper(Dataset[dict[str, torch.Tensor | str | int]]):
    """Wraps ``BraTSSliceDataset`` so ``image`` has 3 channels (SegFormer RGB).

    BraTS provides 4 modalities; we keep *channels* (default: t1c, t2f, t2w)
    and drop the rest.
    """

    def __init__(
        self,
        base_dataset: BraTSSliceDataset,
        channels: tuple[int, ...] = _DEFAULT_CHANNELS,
    ) -> None:
        self._base = base_dataset
        self._channels = list(channels)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        item = self._base[index]
        item["image"] = item["image"][self._channels, :, :]  # type: ignore[index]
        return item


# ── model surgery ────────────────────────────────────────────────────────

def _widen_first_conv(model: torch.nn.Module) -> None:
    """Replace the stage-0 overlap-patch-embed Conv2d to accept 4 input channels.

    Existing RGB weights are copied; the fourth kernel slice is initialised as
    the channel-wise mean of the original three filters.
    """
    pe = model.segformer.stages[0].patch_embeddings  # type: ignore[union-attr]
    old_conv: torch.nn.Conv2d = pe.proj

    new_conv = torch.nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    pe.proj = new_conv


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer + LoRA on BraTS glioma segmentation",
    )

    # model
    parser.add_argument(
        "--model",
        choices=_MODEL_CHOICES,
        default="nvidia/mit-b0",
        help="SegFormer variant",
    )
    parser.add_argument(
        "--use-4channel",
        action="store_true",
        help="Surgically widen first conv to accept all 4 BraTS modalities",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    # data
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit training cases (0=all)")
    parser.add_argument("--include-empty", action="store_true")

    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument(
        "--mixed-precision",
        default="bf16",
        choices=_MIXED_PRECISION_CHOICES,
        help="Mixed precision mode",
    )
    parser.add_argument("--num-workers", type=int, default=0)

    # outputs
    parser.add_argument("--output-dir", default="artifacts/segformer_lora")

    return parser.parse_args()


# ── training / evaluation ────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    grad_accum: int = 1,
) -> float:
    model.train()
    losses: list[float] = []
    pbar = tqdm(
        loader,
        desc="train",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )
    for step, batch in enumerate(pbar):
        image: torch.Tensor = batch["image"]  # type: ignore[assignment]
        target: torch.Tensor = batch["target"]  # type: ignore[assignment]

        with accelerator.autocast():
            outputs = model(pixel_values=image)
            logits: torch.Tensor = outputs.logits
            # SegFormer outputs at 1/4 resolution → upsample to target size
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = dice_bce_loss(logits, target) / grad_accum

        accelerator.backward(loss)
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()) * grad_accum)
        if accelerator.is_local_main_process:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    # drain any leftover gradients from the last incomplete accumulation step
    if len(loader) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    threshold: float = 0.5,
) -> dict:
    """Evaluate on the main process only.

    For multi-GPU we gather logits & targets to rank 0, then compute Dice /
    HD95 serially on a single CPU.  With a 30-case BraTS subset this is
    negligible overhead and avoids dict-gathering complexity.
    """
    model.eval()
    gathered_logits: list[torch.Tensor] = []
    gathered_targets: list[torch.Tensor] = []

    for batch in tqdm(
        loader,
        desc="eval",
        leave=False,
        disable=not accelerator.is_local_main_process,
    ):
        image: torch.Tensor = batch["image"]  # type: ignore[assignment]
        target: torch.Tensor = batch["target"]  # type: ignore[assignment]

        with accelerator.autocast():
            outputs = model(pixel_values=image)
            logits: torch.Tensor = outputs.logits
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Gather across devices so rank 0 sees the full validation set.
        gl = accelerator.gather(logits)
        gt = accelerator.gather(target)
        if accelerator.is_main_process:
            gathered_logits.append(gl.cpu())
            gathered_targets.append(gt.cpu())

    if not accelerator.is_main_process:
        # Non-main ranks return a stub; main processes the final metrics.
        return {"loss": 0.0, "dice": {}, "hd95": {}}

    all_logits = torch.cat(gathered_logits, dim=0)
    all_targets = torch.cat(gathered_targets, dim=0)

    val_loss = float(dice_bce_loss(all_logits, all_targets).item())
    preds = threshold_predictions(all_logits, threshold=threshold)
    truth = all_targets.numpy().astype(np.uint8)

    dice_list: list[dict[str, float]] = []
    hd95_list: list[dict[str, float]] = []
    for pred_np, truth_np in zip(preds, truth, strict=True):
        dice_list.append(dice_per_region(pred_np, truth_np))
        hd95_list.append(hd95_per_region(pred_np, truth_np))

    keys = list(dice_list[0].keys()) if dice_list else []
    avg_dice = {k: float(np.mean([d[k] for d in dice_list])) for k in keys}
    avg_hd95 = {k: float(np.mean([d[k] for d in hd95_list])) for k in keys}

    return {"loss": val_loss, "dice": avg_dice, "hd95": avg_hd95}


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Accelerator ──────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision
        if args.mixed_precision != "no"
        else None,
        gradient_accumulation_steps=args.grad_accum,
    )

    # ── Device info ──────────────────────────────────────────────────────
    dev = get_device()
    dev_name = device_name()
    if accelerator.is_main_process:
        print(f"Device: {dev} ({dev_name})")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"World size: {accelerator.num_processes}")

    # ── Data ─────────────────────────────────────────────────────────────
    cases = discover_cases(args.data_root)
    cases = limit_cases(cases, args.max_cases or None)
    splits = stable_split_cases(cases)

    base_train = BraTSSliceDataset(
        splits["train"],
        include_empty=args.include_empty,
        augment=True,
    )
    base_val = BraTSSliceDataset(
        splits["val"],
        include_empty=False,
        augment=False,
    )

    # Wrap for 3-channel SegFormer input (skip wrapper when using 4ch surgery)
    if args.use_4channel:
        train_dataset = base_train
        val_dataset = base_val
    else:
        train_dataset = _SegFormerDatasetWrapper(base_train)
        val_dataset = _SegFormerDatasetWrapper(base_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(dev != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev != "cpu"),
    )

    if accelerator.is_main_process:
        print(f"Train cases: {len(splits['train'])}, Val cases: {len(splits['val'])}")
        print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print(f"Loading {args.model} …")

    id2label = {0: "WT", 1: "TC", 2: "ET"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Optional: widen first convolution for native 4-channel input.
    if args.use_4channel:
        _widen_first_conv(model)
        if accelerator.is_main_process:
            print("  First patch-embed widened to 4 input channels.")

    # ── PEFT / LoRA ──────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["decode_head"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    # ── Prepare with Accelerator ─────────────────────────────────────────
    # NOTE: we do NOT pass the scheduler to prepare() because
    # CosineAnnealingWarmRestarts is stepped per epoch, not per batch.
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )

    # ── Training loop ────────────────────────────────────────────────────
    csv_path = output_dir / "history.csv"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, accelerator,
            grad_accum=args.grad_accum,
        )
        scheduler.step(epoch - 1)

        val_summary = evaluate(model, val_loader, accelerator)
        val_loss = val_summary["loss"]
        dice_row = val_summary["dice"]
        hd95_row = val_summary["hd95"]

        # Reduce train_loss across processes for consistent logging.
        train_loss_t = torch.tensor([train_loss], device=accelerator.device)
        train_loss = float(accelerator.gather(train_loss_t).mean().item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if accelerator.is_main_process:
            write_header = not csv_path.exists()
            with csv_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow([
                        "epoch", "train_loss", "val_loss",
                        "dice_WT", "dice_TC", "dice_ET",
                        "hd95_WT", "hd95_TC", "hd95_ET",
                    ])
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{dice_row.get('WT', 0):.4f}",
                    f"{dice_row.get('TC', 0):.4f}",
                    f"{dice_row.get('ET', 0):.4f}",
                    f"{hd95_row.get('WT', 0):.2f}",
                    f"{hd95_row.get('TC', 0):.2f}",
                    f"{hd95_row.get('ET', 0):.2f}",
                ])

            print(
                f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
                f"Dice WT:{dice_row.get('WT', 0):.3f} "
                f"TC:{dice_row.get('TC', 0):.3f} "
                f"ET:{dice_row.get('ET', 0):.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_summary

                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(output_dir / "best_model")
                print(f"  >> Saved best LoRA adapter (val_loss={val_loss:.4f})")

        accelerator.wait_for_everyone()

    # ── Save final artefacts (main process) ──────────────────────────────
    if accelerator.is_main_process:
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        (output_dir / "best_metrics.json").write_text(
            json.dumps(best_metrics, indent=2), encoding="utf-8",
        )
        save_loss_curve(history, output_dir / "loss_curve.png")
        print(f"\nDone. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
