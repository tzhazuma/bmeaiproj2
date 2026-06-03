from __future__ import annotations

import argparse
import csv
import json
import multiprocessing

# Fix Python 3.14 multiprocessing semaphore leak
multiprocessing.set_start_method("spawn", force=True)

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def _create_scaler(device: str, enabled: bool) -> torch.amp.GradScaler | None:
    if not enabled:
        return None
    amp_device = device if device in ("cuda", "xpu") else "cpu"
    return torch.amp.GradScaler(amp_device)


def _autocast_ctx(device: str, enabled: bool):
    amp_device = device if device in ("cuda", "xpu") else "cpu"
    return torch.amp.autocast(amp_device, enabled=enabled)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import (
    BraTSMultiSliceDataset,
    BraTSSliceDataset,
    CachedBraTSSliceDataset,
    discover_cases,
    limit_cases,
    load_split_manifest,
    save_split_manifest,
    stable_split_cases,
)
from brats_seg.fusion import (
    CrossModalityGate,
    ModalityAttentionFusion,
    ModalitySpecificEncoder,
    SliceContextFusion,
)
from brats_seg.losses import dice_bce_loss, dice_focal_loss
from brats_seg.metrics import dice_per_region, hd95_per_region, threshold_predictions
from brats_seg.models import (
    UNet2D,
    AttentionUNet2D,
    DeepSupAttentionUNet2D,
    MAUNet2D,
    MedNeXt2D,
    ResEncUNet2D,
    SegResNet2D,
    SwinUNETR2D,
)
from brats_seg.device import get_device
from brats_seg.visualization import save_loss_curve

MODEL_DEEP_SUP = {"resenc_unet", "deep_attention_unet", "maunet"}
MODEL_VAE = {"segresnet"}
MODEL_CHECKPOINT = {"swin_unetr", "mednext", "resenc_unet"}


def get_loss_function(loss_name: str):
    if loss_name == "dice_bce":
        return dice_bce_loss
    if loss_name == "dice_focal":
        return dice_focal_loss
    raise ValueError(f"Unknown loss function: {loss_name}")


def create_backbone(model_name: str, in_channels: int, out_channels: int, use_checkpoint: bool) -> nn.Module:
    """Instantiate a segmentation backbone by name."""
    common_2d = dict(in_channels=in_channels, out_channels=out_channels)

    if model_name == "unet":
        return UNet2D(**common_2d)

    if model_name == "attention_unet":
        return AttentionUNet2D(**common_2d)

    if model_name == "resenc_unet":
        return ResEncUNet2D(**common_2d, use_checkpoint=use_checkpoint)

    if model_name == "deep_attention_unet":
        return DeepSupAttentionUNet2D(**common_2d)

    if model_name == "maunet":
        return MAUNet2D(**common_2d)

    if model_name == "mednext":
        return MedNeXt2D(**common_2d, use_checkpoint=use_checkpoint)

    if model_name == "segresnet":
        return SegResNet2D(**common_2d)

    if model_name == "swin_unetr":
        return SwinUNETR2D(**common_2d, use_checkpoint=use_checkpoint)

    raise ValueError(f"Unknown model: {model_name}")


def create_fusion(fusion_type: str, num_modalities: int = 4, num_slices: int = 3) -> nn.Module:
    """Instantiate a fusion module by type."""
    if fusion_type == "se":
        return ModalityAttentionFusion(num_modalities=num_modalities)
    if fusion_type == "cross":
        return CrossModalityGate(num_modalities=num_modalities)
    if fusion_type == "specific":
        return ModalitySpecificEncoder(num_modalities=num_modalities)
    if fusion_type == "slice":
        return SliceContextFusion(num_slices=num_slices, in_channels=num_modalities)
    raise ValueError(f"Unknown fusion type: {fusion_type}")


def fusion_output_channels(fusion_type: str, multi_slice: int, num_modalities: int = 4) -> int:
    """Return the channel count after a fusion module."""
    if fusion_type in {"se", "cross", ""}:
        return num_modalities * max(multi_slice, 1)
    if fusion_type == "specific":
        return 64
    if fusion_type == "slice":
        return 32
    raise ValueError(f"Unknown fusion type: {fusion_type}")


def compute_deep_sup_loss(
    main: torch.Tensor,
    aux: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    """Weighted deep-supervision loss for 3 auxiliary heads."""
    loss_fn = get_loss_function(loss_name)
    loss = loss_fn(main, target)
    loss += 0.5 * loss_fn(aux[0], target)
    loss += 0.25 * loss_fn(aux[1], target)
    loss += 0.125 * loss_fn(aux[2], target)
    return loss


def compute_vae_loss(
    main: torch.Tensor,
    aux1: torch.Tensor,
    aux2: torch.Tensor,
    vae_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    raw_input: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    """SegResNet2D loss: segmentation loss + deep-sup + KL + VAE reconstruction."""
    loss_fn = get_loss_function(loss_name)
    loss = loss_fn(main, target)
    loss += 0.5 * loss_fn(aux1, target)
    loss += 0.3 * loss_fn(aux2, target)

    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
    recon = F.mse_loss(vae_recon, raw_input[:, :4, :, :])
    loss += 0.1 * kl + 0.1 * recon
    return loss


def train_one_epoch(
    backbone: nn.Module,
    fusion: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: str,
    model_name: str,
    loss_name: str,
) -> float:
    """Run a single training epoch. Returns average loss."""
    backbone.train()
    if fusion is not None:
        fusion.train()

    is_deep_sup = model_name in MODEL_DEEP_SUP
    is_vae = model_name in MODEL_VAE
    use_amp = scaler is not None
    losses: list[float] = []

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        image = batch["image"].float().to(device)
        target = batch["target"].float().to(device)

        # Save raw image before fusion (needed for VAE reconstruction loss)
        image_raw = image

        if fusion is not None:
            image = fusion(image)

        with _autocast_ctx(device, use_amp):
            if is_vae:
                main, aux1, aux2, vae_recon, mu, log_var = backbone(image)
                loss = compute_vae_loss(main, aux1, aux2, vae_recon, mu, log_var, target, image_raw, loss_name)
            elif is_deep_sup:
                main, aux1, aux2, aux3 = backbone(image, return_aux=True)
                loss = compute_deep_sup_loss(main, (aux1, aux2, aux3), target, loss_name)
            else:
                output = backbone(image)
                loss = get_loss_function(loss_name)(output, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    fusion: nn.Module | None,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    amp_enabled: bool = False,
    loss_name: str = "dice_bce",
) -> dict:
    """Run validation: compute loss, dice, hd95."""
    backbone.eval()
    if fusion is not None:
        fusion.eval()

    losses: list[float] = []
    dice_list: list[dict[str, float]] = []
    hd95_list: list[dict[str, float]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].float().to(device)
        target = batch["target"].float().to(device)

        if fusion is not None:
            image = fusion(image)

        with _autocast_ctx(device, amp_enabled):
            logits = backbone(image)
            loss = get_loss_function(loss_name)(logits, target)

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
    parser = argparse.ArgumentParser(description="Unified BraTS training script")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="artifacts/run")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "unet",
            "attention_unet",
            "resenc_unet",
            "deep_attention_unet",
            "maunet",
            "mednext",
            "segresnet",
            "swin_unetr",
        ],
        help="Segmentation model architecture",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--multi-slice", type=int, default=0, help="Number of slices for 2.5D input (0 = 2D only)")
    parser.add_argument(
        "--fusion",
        default="",
        choices=["", "se", "cross", "specific", "slice"],
        help="Modality fusion module",
    )
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed-precision training")
    parser.add_argument(
        "--loss",
        default="dice_bce",
        choices=["dice_bce", "dice_focal"],
        help="Segmentation loss used for training and validation loss tracking",
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument(
        "--lr-schedule",
        default="cosine",
        choices=["plateau", "cosine"],
        help="Learning rate schedule",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of training cases (0 = all)")
    parser.add_argument("--splits", default="", help="Path to a pre-saved splits manifest")
    parser.add_argument("--include-empty", action="store_true", help="Include empty slices in training")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", default="/tmp/brats_preprocessed", help="Use preprocessed cache dir for fast loading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    else:
        cases = discover_cases(args.data_root)
        cases = limit_cases(cases, args.max_cases or None)
        splits = stable_split_cases(cases)
        save_split_manifest(splits, output_dir / "splits.json")

    multi_slice = args.multi_slice
    include_empty = args.include_empty

    if multi_slice > 0:
        train_dataset = BraTSMultiSliceDataset(
            splits["train"], num_slices=multi_slice, include_empty=include_empty, augment=True
        )
        val_dataset = BraTSMultiSliceDataset(
            splits["val"], num_slices=multi_slice, include_empty=False, augment=False
        )
    else:
        cache_dir = args.cache_dir
        use_cache = Path(cache_dir).exists() if cache_dir else False
        if use_cache:
            train_dataset = CachedBraTSSliceDataset(
                splits["train"], cache_dir=cache_dir, include_empty=include_empty, augment=True, cache_size=8
            )
            val_dataset = CachedBraTSSliceDataset(
                splits["val"], cache_dir=cache_dir, include_empty=False, augment=False, cache_size=8
            )
        else:
            train_dataset = BraTSSliceDataset(splits["train"], include_empty=include_empty, augment=True)
            val_dataset = BraTSSliceDataset(splits["val"], include_empty=False, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Train cases: {len(splits['train'])}, Val cases: {len(splits['val'])}")
    print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

    device = get_device()
    print(f"Device: {device}")

    if args.fusion and multi_slice > 0 and args.fusion != "slice":
        raise ValueError(f"Fusion type '{args.fusion}' is only supported for single-slice (4-channel) input, not multi-slice.")
    if args.fusion == "slice" and multi_slice <= 0:
        raise ValueError("Fusion type 'slice' requires --multi-slice > 0.")

    num_modalities = 4
    in_channels = num_modalities * max(multi_slice, 1)

    fusion_module: nn.Module | None = None
    backbone_in_channels = in_channels

    if args.fusion:
        fusion_module = create_fusion(args.fusion, num_modalities=num_modalities, num_slices=multi_slice or 3)
        backbone_in_channels = fusion_output_channels(args.fusion, multi_slice, num_modalities)

    use_ckpt = not args.no_checkpoint and args.model in MODEL_CHECKPOINT
    backbone = create_backbone(args.model, backbone_in_channels, out_channels=3, use_checkpoint=use_ckpt)

    backbone.to(device)
    if fusion_module is not None:
        fusion_module.to(device)

    params = list(backbone.parameters())
    if fusion_module is not None:
        params += list(fusion_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.lr_schedule == "cosine":
        scheduler: ReduceLROnPlateau | CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler: torch.amp.GradScaler | None = _create_scaler(device, args.amp)
    use_amp = args.amp

    csv_path = output_dir / "history.csv"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            backbone=backbone,
            fusion=fusion_module,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            model_name=args.model,
            loss_name=args.loss,
        )

        val_summary = evaluate(
            backbone=backbone,
            fusion=fusion_module,
            loader=val_loader,
            device=device,
            amp_enabled=use_amp,
            loss_name=args.loss,
        )
        val_loss = val_summary["loss"]

        if isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(epoch - 1)
        else:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        dice_row = val_summary["dice"]
        hd95_row = val_summary["hd95"]
        write_csv = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if write_csv:
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
                "backbone": backbone.state_dict(),
                "fusion": fusion_module.state_dict() if fusion_module is not None else None,
                "epoch": epoch,
                "val_loss": val_loss,
                "model": args.model,
                "loss": args.loss,
                "fusion_type": args.fusion,
                "multi_slice": multi_slice,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  >> Saved best model (val_loss={val_loss:.4f})")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    save_loss_curve(history, output_dir / "loss_curve.png")
    print(f"\nDone. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
