"""Fine-tune a SAM-like pretrained model on BraTS glioma segmentation.

Supports two modes:

- ``--source sam`` — loads Meta's SAM (Segment Anything Model) from a
  downloaded checkpoint.  Requires ``pip install brats-seg[sam]``.
- ``--source vit`` — uses a pure-PyTorch ViT-inspired encoder (random init,
  no external dependency).  Useful for testing the pipeline.

Usage::

    # Standalone mode (random init, no download needed)
    python scripts/finetune_sam.py --source vit --epochs 5

    # Full SAM mode (requires checkpoint from Meta)
    python scripts/finetune_sam.py --source sam \\
        --checkpoint /path/to/sam_vit_b_01ec64.pth --epochs 15

    # With a custom data root and output directory
    python scripts/finetune_sam.py --source vit \\
        --data-root /path/to/brats2023 --output-dir artifacts/sam_finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import multiprocessing
from pathlib import Path

# Fix Python 3.14 multiprocessing semaphore leak: use spawn context
multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import BraTSSliceDataset, CachedBraTSSliceDataset, discover_cases, limit_cases, stable_split_cases
from brats_seg.device import device_name, get_device
from brats_seg.losses import dice_bce_loss
from brats_seg.metrics import dice_per_region, hd95_per_region, threshold_predictions
from brats_seg.models.sam_adapter import SAMAdapterConfig, create_sam_adapter
from brats_seg.visualization import save_loss_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM-like model on BraTS")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="artifacts/sam_finetune")
    parser.add_argument("--source", choices=("sam", "vit"), default="vit", help="Model source")
    parser.add_argument("--checkpoint", default="", help="Path to SAM checkpoint")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Unfreeze ViT encoder for end-to-end fine-tuning (default: frozen)")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit training cases (0 = all)")
    parser.add_argument("--splits", default="", help="Path to pre-saved splits manifest")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", default="", help="Use preprocessed cache dir for fast loading (skips on-the-fly NIfTI preprocessing)")
    parser.add_argument("--amp", action="store_true", default=True, help="Use bf16 mixed precision via torch.amp (default: on)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--gpu", type=str, default="", help="Select specific GPU(s) via CUDA_VISIBLE_DEVICES (e.g. --gpu 0 or --gpu 0,1)")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint (.pt file)")
    args = parser.parse_args()
    args.freeze_encoder = not args.unfreeze_encoder
    return args


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
) -> float:
    model.train()
    losses: list[float] = []
    pbar = tqdm(loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(pbar):
        image = batch["image"].float().to(device)
        target = batch["target"].float().to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _null_context():
            logits = model(image)
            loss = dice_bce_loss(logits, target)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item() * grad_accum_steps))
        postfix = {"loss": f"{losses[-1]:.4f}"}
        if device.startswith("cuda"):
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            postfix["alloc"] = f"{alloc:.0f}M"
            postfix["cache"] = f"{reserv:.0f}M"
        pbar.set_postfix(**postfix)
    return float(np.mean(losses)) if losses else 0.0


class _NullContext:
    """Context manager that does nothing — used as a no-op alternative to autocast."""

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


def _null_context() -> _NullContext:
    return _NullContext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> dict:
    model.eval()
    losses: list[float] = []
    dice_list: list[dict[str, float]] = []
    hd95_list: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].float().to(device)
        target = batch["target"].float().to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _null_context():
            logits = model(image)
            loss = dice_bce_loss(logits, target)
        losses.append(float(loss.item()))
        preds = threshold_predictions(logits, threshold=threshold)
        truth = target.cpu().numpy().astype(np.uint8)
        for pred_np, truth_np in zip(preds, truth, strict=True):
            dice_list.append(dice_per_region(pred_np, truth_np))
            hd95_list.append(hd95_per_region(pred_np, truth_np))
    keys = dice_list[0].keys() if dice_list else []
    avg_dice = {k: float(np.mean([d[k] for d in dice_list])) for k in keys}
    avg_hd95 = {k: float(np.mean([d[k] for d in hd95_list])) for k in keys}
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dice": avg_dice,
        "hd95": avg_hd95,
    }


def main() -> None:
    args = parse_args()

    # ── GPU selection (must happen before any CUDA context is created) ──
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    cases = discover_cases(args.data_root)
    cases = limit_cases(cases, args.max_cases or None)
    splits = stable_split_cases(cases)

    use_cache = args.cache_dir and Path(args.cache_dir).exists()
    if use_cache:
        print(f"Using preprocessed cache: {args.cache_dir}")
        train_dataset = CachedBraTSSliceDataset(splits["train"], cache_dir=args.cache_dir, include_empty=args.include_empty, augment=True)
        val_dataset = CachedBraTSSliceDataset(splits["val"], cache_dir=args.cache_dir, include_empty=False, augment=False)
    else:
        train_dataset = BraTSSliceDataset(splits["train"], include_empty=args.include_empty, augment=True)
        val_dataset = BraTSSliceDataset(splits["val"], include_empty=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train cases: {len(splits['train'])}, Val cases: {len(splits['val'])}")
    print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}  |  Grad accum: {args.grad_accum_steps}  |  Effective batch: {args.batch_size * args.grad_accum_steps}")

    # ── Device ──────────────────────────────────────────────────────────
    device = get_device()
    dev_name = device_name()
    print(f"Device: {device} ({dev_name})")

    use_amp = args.amp and device.startswith("cuda")
    if args.amp and not device.startswith("cuda"):
        print("[warn] --amp requires CUDA; mixed precision disabled")
    if use_amp:
        print("Mixed precision: bf16 (torch.amp.autocast)")

    # ── AMP scaler (bf16: scaler is a no-op for underflow but forward-compatible) ──
    scaler: torch.amp.GradScaler | None = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    # ── Model ───────────────────────────────────────────────────────────
    config = SAMAdapterConfig(
        source=args.source,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint if args.source == "sam" else "",
        in_channels=4,
        out_channels=3,
        freeze_encoder=args.freeze_encoder,
    )
    model = create_sam_adapter(config)
    model.to(device)

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed (epoch {start_epoch - 1})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # ── Gradient checkpointing on encoder ───────────────────────────────
    if hasattr(model, "encoder") and device.startswith("cuda"):
        from torch.utils.checkpoint import checkpoint as _checkpoint

        _orig_forward = model.encoder.forward

        def _ckpt_forward(x: torch.Tensor) -> torch.Tensor:
            return _checkpoint(_orig_forward, x, use_reentrant=False)

        model.encoder.forward = _ckpt_forward  # type: ignore[method-assign]
        print("Gradient checkpointing enabled on encoder")

    # ── Optimiser ───────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    if args.resume and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint")
        except Exception:
            print("Could not load optimizer state, using fresh optimizer")

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ── Training loop ───────────────────────────────────────────────────
    csv_path = output_dir / "history.csv"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_metrics: dict = {}

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_summary = evaluate(model, val_loader, device, use_amp=use_amp)
        val_loss = val_summary["loss"]
        scheduler.step(epoch - 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        dice_row = val_summary["dice"]
        hd95_row = val_summary["hd95"]
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
            f"Dice WT:{dice_row.get('WT', 0):.3f} TC:{dice_row.get('TC', 0):.3f} ET:{dice_row.get('ET', 0):.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_summary
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "source": args.source,
                    "model_type": args.model_type,
                    "in_channels": 4,
                    "out_channels": 3,
                    "freeze_encoder": args.freeze_encoder,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  >> Saved best model (val_loss={val_loss:.4f})")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    save_loss_curve(history, output_dir / "loss_curve.png")
    print(f"\nDone. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
