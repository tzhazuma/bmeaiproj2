from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .constants import DEFAULT_DATA_ROOT, MODALITIES
from .preprocessing import (
    BoundingBox,
    compute_foreground_bbox,
    crop_mask,
    crop_volumes,
    pad_or_crop_2d,
    segmentation_to_regions,
    zscore_normalize,
)


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    case_dir: Path
    modality_paths: dict[str, Path]
    seg_path: Path


class ProcessedCase(TypedDict):
    image: np.ndarray
    seg: np.ndarray
    regions: np.ndarray
    bbox: BoundingBox


class SliceRecord(TypedDict):
    case_id: str
    slice_index: int
    positive: bool


def discover_cases(data_root: str | Path = DEFAULT_DATA_ROOT) -> list[BraTSCase]:
    root = Path(data_root)
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("BraTS-GLI-"))
    cases: list[BraTSCase] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        modality_paths = {modality: case_dir / f"{case_id}-{modality}.nii" for modality in MODALITIES}
        seg_path = case_dir / f"{case_id}-seg.nii"
        if not all(path.exists() for path in modality_paths.values()) or not seg_path.exists():
            continue
        cases.append(BraTSCase(case_id=case_id, case_dir=case_dir, modality_paths=modality_paths, seg_path=seg_path))
    return cases


def stable_split_cases(
    cases: list[BraTSCase],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[BraTSCase]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must define a valid three-way split")
    ordered = sorted(cases, key=lambda case: hashlib.sha1(case.case_id.encode("utf-8")).hexdigest())
    total = len(ordered)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def save_split_manifest(splits: dict[str, list[BraTSCase]], path: str | Path) -> None:
    payload = {split: [case.case_id for case in cases] for split, cases in splits.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(data_root: str | Path, path: str | Path) -> dict[str, list[BraTSCase]]:
    cases = {case.case_id: case for case in discover_cases(data_root)}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {split: [cases[case_id] for case_id in case_ids] for split, case_ids in payload.items()}


def load_case_arrays(case: BraTSCase) -> tuple[np.ndarray, np.ndarray]:
    volumes = []
    for modality in MODALITIES:
        image = cast(nib.Nifti1Image, nib.load(str(case.modality_paths[modality])))
        array = image.get_fdata(dtype=np.float32)
        volumes.append(np.transpose(array, (2, 0, 1)))
    seg_image = cast(nib.Nifti1Image, nib.load(str(case.seg_path)))
    seg = seg_image.get_fdata(dtype=np.float32)
    seg = np.transpose(seg, (2, 0, 1))
    return np.stack(volumes, axis=0), seg


def preprocess_case(case: BraTSCase, crop_margin: int = 5) -> ProcessedCase:
    volumes, seg = load_case_arrays(case)
    normalized = np.stack([zscore_normalize(channel) for channel in volumes], axis=0)
    bbox = compute_foreground_bbox(normalized, margin=crop_margin)
    cropped_volumes = crop_volumes(normalized, bbox)
    cropped_seg = crop_mask(seg, bbox)
    return {
        "image": cropped_volumes.astype(np.float32),
        "seg": cropped_seg.astype(np.int16),
        "regions": segmentation_to_regions(cropped_seg),
        "bbox": bbox,
    }


def build_slice_records(cases: list[BraTSCase], include_empty: bool = False) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    for case in cases:
        processed = preprocess_case(case)
        regions = processed["regions"]
        positive_slices = np.any(regions > 0, axis=(0, 2, 3))
        for slice_index, is_positive in enumerate(positive_slices.tolist()):
            if include_empty or is_positive:
                records.append({"case_id": case.case_id, "slice_index": slice_index, "positive": bool(is_positive)})
    return records


def limit_cases(cases: list[BraTSCase], max_cases: int | None = None) -> list[BraTSCase]:
    if max_cases is None or max_cases <= 0:
        return cases
    return cases[:max_cases]


class SliceAugmentor:
    def __init__(self, enabled: bool = True, noise_std: float = 0.05) -> None:
        self.enabled = enabled
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return image, target
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=-1).copy()
            target = np.flip(target, axis=-1).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=-2).copy()
            target = np.flip(target, axis=-2).copy()
        k = int(np.random.randint(0, 4))
        image = np.rot90(image, k, axes=(-2, -1)).copy()
        target = np.rot90(target, k, axes=(-2, -1)).copy()
        if np.random.rand() < 0.5:
            noise = np.random.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
            image = image + noise
        scale = float(np.random.uniform(0.95, 1.05))
        if abs(scale - 1.0) > 1e-3:
            from scipy.ndimage import zoom

            zoom_factors = (1.0, scale, scale)
            image = np.asarray(zoom(image, zoom_factors, order=1), dtype=np.float32)
            target = np.asarray(zoom(target, zoom_factors, order=0), dtype=np.float32)
            from .preprocessing import pad_or_crop_2d

            image = pad_or_crop_2d(image, target_shape=(160, 160))
            target = pad_or_crop_2d(target, target_shape=(160, 160))
        return image.astype(np.float32), target.astype(np.float32)


class BraTSSliceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cases: list[BraTSCase],
        include_empty: bool = False,
        augment: bool = False,
        cache_size: int = 2,
        target_shape: tuple[int, int] = (160, 160),
    ) -> None:
        self.cases = {case.case_id: case for case in cases}
        self.records = build_slice_records(cases, include_empty=include_empty)
        self.augmentor = SliceAugmentor(enabled=augment)
        self.cache: OrderedDict[str, ProcessedCase] = OrderedDict()
        self.cache_size = cache_size
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.records)

    def _get_case(self, case_id: str) -> ProcessedCase:
        if case_id in self.cache:
            cached = self.cache.pop(case_id)
            self.cache[case_id] = cached
            return cached
        processed = preprocess_case(self.cases[case_id])
        self.cache[case_id] = processed
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return processed

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        case_id = record["case_id"]
        slice_index = record["slice_index"]
        processed = self._get_case(case_id)
        image = processed["image"][:, slice_index, :, :]
        target = processed["regions"][:, slice_index, :, :]
        image = pad_or_crop_2d(image, self.target_shape)
        target = pad_or_crop_2d(target, self.target_shape)
        image, target = self.augmentor(image, target)
        return {
            "image": Tensor(image.tolist()),
            "target": Tensor(target.tolist()),
            "case_id": case_id,
            "slice_index": slice_index,
        }


def summarize_dataset(cases: list[BraTSCase]) -> dict[str, object]:
    suffix_counts: dict[str, int] = {}
    for case in cases:
        suffix = case.case_id.rsplit("-", 1)[-1]
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "num_cases": len(cases),
        "modalities": list(MODALITIES),
        "suffix_counts": suffix_counts,
    }
