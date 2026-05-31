from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import discover_cases, limit_cases, load_split_manifest, preprocess_case, stable_split_cases
from brats_seg.inference import predict_case_regions
from brats_seg.models import AttentionUNet2D, UNet2D
from brats_seg.training import create_device
from brats_seg.visualization import choose_representative_slice, save_prediction_figure


def load_model_state(checkpoint_path: str | Path, device: str) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/prediction_examples")
    parser.add_argument("--model", choices=("unet", "attention_unet"), default="unet")
    parser.add_argument("--splits", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    else:
        cases = limit_cases(discover_cases(args.data_root), args.max_cases or None)
        splits = stable_split_cases(cases)

    cases = splits[args.split][: args.num_examples]
    if not cases:
        raise ValueError(f"No cases found in split: {args.split}")

    device = create_device()
    model = UNet2D() if args.model == "unet" else AttentionUNet2D()
    model.load_state_dict(load_model_state(args.checkpoint, device))
    model.to(device)

    for case in cases:
        logging.info("Generating prediction figure for %s", case.case_id)
        processed = preprocess_case(case)
        result = predict_case_regions(
            model,
            processed["image"],
            processed["regions"],
            device,
            case_id=case.case_id,
            threshold=args.threshold,
        )
        slice_index = choose_representative_slice(result.target)
        save_prediction_figure(
            result.image,
            result.target,
            result.prediction,
            output_dir / f"{case.case_id}_prediction.png",
            slice_index,
        )

    logging.info("Saved %d prediction figures to %s", len(cases), output_dir)


if __name__ == "__main__":
    main()
