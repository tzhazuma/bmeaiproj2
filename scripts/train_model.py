from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import BraTSSliceDataset, discover_cases, limit_cases, load_split_manifest, save_split_manifest, stable_split_cases
from brats_seg.models import AttentionUNet2D, UNet2D
from brats_seg.training import TrainConfig, fit
from brats_seg.visualization import save_loss_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--model", choices=("unet", "attention_unet"), default="unet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-cases", type=int, default=0)
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

    train_dataset = BraTSSliceDataset(splits["train"], include_empty=args.include_empty, augment=True)
    val_dataset = BraTSSliceDataset(splits["val"], include_empty=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet2D() if args.model == "unet" else AttentionUNet2D()
    history = fit(model, train_loader, val_loader, output_dir, TrainConfig(epochs=args.epochs, batch_size=args.batch_size))
    save_loss_curve(history, output_dir / "loss_curve.png")


if __name__ == "__main__":
    main()
