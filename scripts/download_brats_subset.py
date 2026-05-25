#!/usr/bin/env python3
"""Download a 30-case subset of BraTS 2023 GLI from HuggingFace.

Repo: Angelou0516/brats2023-gli-dataset (dataset type)
Source layout: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/{case_id}/{files}.nii.gz
Target layout: data/brats2023/{case_id}/{case_id}-{modality}.nii.gz
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Angelou0516/brats2023-gli-dataset"
REPO_TYPE = "dataset"
SOURCE_PREFIX = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
MODALITIES = ("t1c", "t1n", "t2f", "t2w")
NUM_CASES = 30


def main() -> None:
    target_root = Path("/home/azuma/bmeaiproj2/data/brats2023")
    target_root.mkdir(parents=True, exist_ok=True)

    print("=== Listing repo files ===", flush=True)
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    nii_files = [f for f in all_files if f.endswith(".nii.gz") and f.startswith(SOURCE_PREFIX)]
    print(f"Found {len(nii_files)} .nii.gz files in repo", flush=True)

    # Group by case ID
    case_files: dict[str, list[str]] = {}
    for f in nii_files:
        m = re.search(r"BraTS-GLI-\d{5}-\d{3}", f)
        if m:
            case_id = m.group()
            case_files.setdefault(case_id, []).append(f)

    print(f"Found {len(case_files)} unique cases", flush=True)

    # Sort and pick first N
    sorted_cases = sorted(case_files.keys())[:NUM_CASES]
    print(f"Will download {len(sorted_cases)} cases ({len(sorted_cases) * 5} files)", flush=True)

    success = 0
    errors = 0

    for i, case_id in enumerate(sorted_cases, 1):
        case_dir = target_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        for repo_path in case_files[case_id]:
            filename = Path(repo_path).name  # e.g., BraTS-GLI-00000-000-t1c.nii.gz
            target_path = case_dir / filename

            if target_path.exists():
                print(f"[{i}/{NUM_CASES}] {case_id} - {filename} (already exists, skipping)", flush=True)
                success += 1
                continue

            try:
                print(f"[{i}/{NUM_CASES}] {case_id} - downloading {filename} ...", flush=True)
                cached = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_path,
                    repo_type=REPO_TYPE,
                )
                shutil.copy2(cached, target_path)
                success += 1
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                errors += 1

    print(f"\n=== Done: {success} succeeded, {errors} errors ===", flush=True)

    # Verify at least 1 case
    print("\n=== Verification ===", flush=True)
    ok, bad = 0, 0
    for case_id in sorted_cases:
        case_dir = target_root / case_id
        expected = [f"{case_id}-{m}.nii.gz" for m in MODALITIES] + [f"{case_id}-seg.nii.gz"]
        missing = [e for e in expected if not (case_dir / e).exists()]
        sizes = []
        for e in expected:
            p = case_dir / e
            if p.exists():
                sz_mb = p.stat().st_size / (1024 * 1024)
                sizes.append(f"{e}={sz_mb:.1f}MB")
        if missing:
            print(f"  {case_id}: MISSING {missing}", flush=True)
            bad += 1
        else:
            print(f"  {case_id}: OK ({', '.join(sizes)})", flush=True)
            ok += 1
        if ok >= 3 and bad == 0:
            print(f"  ... (verified first {ok} cases, all good)")
            break

    print(f"\nVerified: {ok} complete, {bad} incomplete", flush=True)
    if bad > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
