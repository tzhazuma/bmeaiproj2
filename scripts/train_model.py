from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from brats_seg.constants import DEFAULT_DATA_ROOT, MODALITIES, REGION_NAMES
from brats_seg.data import (
    CachedBraTSSliceDataset,
    CachedRandomAugmentedSliceDataset,
    discover_cases,
    limit_cases,
    load_split_manifest,
    save_split_manifest,
    stable_split_cases,
)
from brats_seg.device import get_device
from brats_seg.models import AttentionUNet2D, RegionModalityAttentionUNet2D, UNet2D
from brats_seg.training import TrainConfig, fit
from brats_seg.visualization import save_loss_curve


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--model", choices=("unet", "attention_unet", "modality_attention_unet"), default="unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128) # batch size should be large enough to ensure stable training, but notice the GPU memory limit.
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--aug-samples-per-slice", type=int, default=1)
    parser.add_argument("--loss", choices=("dice_bce", "dice_focal"), default="dice_bce")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir.parent / "preprocessed_cache"

    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    elif (cache_dir / "splits.json").exists():
        splits = load_split_manifest(args.data_root, cache_dir / "splits.json")
    else:
        cases = discover_cases(args.data_root)
        cases = limit_cases(cases, args.max_cases or None)
        splits = stable_split_cases(cases) # 0.7: 0.15：0.15
        save_split_manifest(splits, output_dir / f"{args.model}_splits.json")

    logging.info("Using preprocessed cache at %s", cache_dir)

    train_dataset = CachedRandomAugmentedSliceDataset(
        splits["train"],
        cache_dir=cache_dir,
        include_empty=args.include_empty,
        aug_samples_per_slice=args.aug_samples_per_slice,
    ) # keep empty slices in training set or not
    val_dataset = CachedBraTSSliceDataset(
        splits["val"],
        cache_dir=cache_dir,
        include_empty=False,
        augment=False,
    ) # actual ratio between train and val set may change after augmentation.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "unet":
        model = UNet2D()
    elif args.model == "attention_unet":
        model = AttentionUNet2D()
    else:
        model = RegionModalityAttentionUNet2D()
    history = fit(
        model,
        train_loader,
        val_loader,
        output_dir,
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, loss=args.loss),
    )
    save_loss_curve(history, output_dir / "loss_curve.png")
    if isinstance(model, RegionModalityAttentionUNet2D):
        attention = model.modality_attention().detach().cpu().numpy()
        payload = {
            region: {modality: float(attention[region_index, modality_index]) for modality_index, modality in enumerate(MODALITIES)}
            for region_index, region in enumerate(REGION_NAMES)
        }
        (output_dir / "learned_modality_attention.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
