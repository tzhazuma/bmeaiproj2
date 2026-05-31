from __future__ import annotations

import argparse
import logging
from pathlib import Path

from brats_seg.constants import DEFAULT_DATA_ROOT
from brats_seg.data import (
    build_case_slice_index_map,
    discover_cases,
    limit_cases,
    load_split_manifest,
    save_processed_cache,
    save_split_manifest,
    stable_split_cases,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", default="artifacts/preprocessed_cache")
    parser.add_argument("--splits", default="")
    parser.add_argument("--split-output", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--slices-per-case", type=int, default=0)
    parser.add_argument("--empty-slice-fraction", type=float, default=0.0)
    parser.add_argument("--sample-seed", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        splits = load_split_manifest(args.data_root, args.splits)
    else:
        cases = discover_cases(args.data_root)
        cases = limit_cases(cases, args.max_cases or None)
        splits = stable_split_cases(cases)
        split_output = Path(args.split_output) if args.split_output else cache_dir / "splits.json"
        split_output.parent.mkdir(parents=True, exist_ok=True)
        save_split_manifest(splits, split_output)
        logging.info("Saved split manifest to %s", split_output)

    all_cases = splits["train"] + splits["val"] + splits["test"]
    slice_indices_by_case = build_case_slice_index_map(
        all_cases,
        slices_per_case=args.slices_per_case,
        empty_fraction=args.empty_slice_fraction,
        seed=args.sample_seed,
    )
    if slice_indices_by_case is not None:
        logging.info(
            "Sampling %d raw slices per case before preprocessing, empty slice fraction %.2f",
            args.slices_per_case,
            args.empty_slice_fraction,
        )
    logging.info("Writing preprocessed case cache for %d cases to %s", len(all_cases), cache_dir / "cases")
    save_processed_cache(
        all_cases,
        cache_dir,
        slice_indices_by_case=slice_indices_by_case,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
