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
    ordered = sorted(cases, key=lambda case: hashlib.new("sha1", case.case_id.encode("utf-8")).hexdigest())
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


def load_case_arrays(case: BraTSCase, slice_indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    volumes = []
    for modality in MODALITIES:
        image = cast(nib.Nifti1Image, nib.load(str(case.modality_paths[modality])))
        array = image.get_fdata(dtype=np.float32)
        if slice_indices is not None:
            array = array[:, :, slice_indices]
        volumes.append(np.transpose(array, (2, 0, 1)))
    seg_image = cast(nib.Nifti1Image, nib.load(str(case.seg_path)))
    seg = seg_image.get_fdata(dtype=np.float32)
    if slice_indices is not None:
        seg = seg[:, :, slice_indices]
    seg = np.transpose(seg, (2, 0, 1))
    return np.stack(volumes, axis=0), seg


def preprocess_case(case: BraTSCase, crop_margin: int = 5, slice_indices: list[int] | None = None) -> ProcessedCase:  # Here is the preprocess function pipeline
    volumes, seg = load_case_arrays(case, slice_indices=slice_indices)
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


def _processed_cache_path(cache_dir: str | Path, case_id: str) -> Path:
    return Path(cache_dir) / "cases" / f"{case_id}.npz"


def sample_case_slice_indices(
    case: BraTSCase,
    slices_per_case: int,
    empty_fraction: float = 0.0,
    seed: int = 2024,
) -> list[int] | None:
    if slices_per_case <= 0:
        return None

    seg_image = cast(nib.Nifti1Image, nib.load(str(case.seg_path)))
    seg = seg_image.get_fdata(dtype=np.float32)
    total_slices = seg.shape[2]
    empty_fraction = min(max(empty_fraction, 0.0), 1.0)
    positive = np.any(seg > 0, axis=(0, 1))
    positive_indices = np.flatnonzero(positive)
    empty_indices = np.flatnonzero(~positive)

    rng_seed = int(hashlib.new("sha1", f"{case.case_id}-{seed}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    num_slices = min(slices_per_case, total_slices)
    num_empty = min(int(round(num_slices * empty_fraction)), len(empty_indices))
    num_positive = min(num_slices - num_empty, len(positive_indices))

    selected: list[int] = []
    if num_positive > 0:
        selected.extend(rng.choice(positive_indices, size=num_positive, replace=False).tolist())
    remaining = num_slices - len(selected)
    if remaining > 0 and len(empty_indices) > 0:
        selected.extend(rng.choice(empty_indices, size=min(remaining, len(empty_indices)), replace=False).tolist())
    remaining = num_slices - len(selected)
    if remaining > 0:
        available = np.setdiff1d(np.arange(total_slices), np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())

    return sorted(int(index) for index in selected)


def build_case_slice_index_map(
    cases: list[BraTSCase],
    slices_per_case: int,
    empty_fraction: float = 0.0,
    seed: int = 2024,
) -> dict[str, list[int]] | None:
    if slices_per_case <= 0:
        return None
    return {
        case.case_id: sample_case_slice_indices(
            case,
            slices_per_case=slices_per_case,
            empty_fraction=empty_fraction,
            seed=seed,
        )
        or []
        for case in cases
    }


def save_processed_cache(
    cases: list[BraTSCase],
    cache_dir: str | Path,
    slice_indices_by_case: dict[str, list[int]] | None = None,
    overwrite: bool = False,
) -> None:
    cache_root = Path(cache_dir)
    cases_dir = cache_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cache_path = _processed_cache_path(cache_root, case.case_id)
        if cache_path.exists() and not overwrite:
            continue
        slice_indices = slice_indices_by_case.get(case.case_id) if slice_indices_by_case else None
        processed = preprocess_case(case, slice_indices=slice_indices)
        np.savez_compressed(
            cache_path,
            image=processed["image"].astype(np.float32),
            seg=processed["seg"].astype(np.int16),
            regions=processed["regions"].astype(np.float32),
            original_slice_indices=np.asarray(slice_indices or [], dtype=np.int16),
        )


def load_processed_cache(cache_dir: str | Path, case_id: str) -> ProcessedCase:
    cache_path = _processed_cache_path(cache_dir, case_id)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing preprocessed cache for {case_id}: {cache_path}")
    data = np.load(str(cache_path))
    processed: ProcessedCase = {
        "image": data["image"].astype(np.float32),
        "seg": data["seg"].astype(np.int16),
        "regions": data["regions"].astype(np.float32),
        "bbox": BoundingBox(0, 0, 0, 0, 0, 0),
    }
    data.close()
    return processed


def build_slice_records_from_cache(
    cases: list[BraTSCase],
    cache_dir: str | Path,
    include_empty: bool = False,
) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    for case in cases:
        processed = load_processed_cache(cache_dir, case.case_id)
        regions = processed["regions"]
        positive_slices = np.any(regions > 0, axis=(0, 2, 3))
        for slice_index, is_positive in enumerate(positive_slices.tolist()):
            if include_empty or is_positive:
                records.append({"case_id": case.case_id, "slice_index": slice_index, "positive": bool(is_positive)})
    return records


def validate_processed_cache(cases: list[BraTSCase], cache_dir: str | Path) -> None:
    missing = [_processed_cache_path(cache_dir, case.case_id) for case in cases if not _processed_cache_path(cache_dir, case.case_id).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} preprocessed cache files. Run scripts/prepare_cache.py first. First missing file: {missing[0]}")


def limit_cases(cases: list[BraTSCase], max_cases: int | None = None) -> list[BraTSCase]:
    if max_cases is None or max_cases <= 0:
        return cases
    return cases[:max_cases]


class SliceAugmentor:
    def __init__(self, enabled: bool = True, noise_std: float = 0.05) -> None:
        self.enabled = enabled
        self.noise_std = noise_std
        self.operations = ("identity", "flip", "rotate", "noise")

    def __call__(self, image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return image, target

        operation = self.operations[int(np.random.randint(0, len(self.operations)))]
        if operation == "identity":
            return image.astype(np.float32), target.astype(np.float32)

        if operation == "flip":
            axis = -1 if np.random.rand() < 0.5 else -2
            image = np.flip(image, axis=axis).copy()
            target = np.flip(target, axis=axis).copy()
        elif operation == "rotate":
            k = int(np.random.randint(1, 4))
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            target = np.rot90(target, k, axes=(-2, -1)).copy()
        elif operation == "noise":
            noise = np.random.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
            image = image + noise
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
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float(),
            "case_id": case_id,
            "slice_index": slice_index,
        }


class CachedBraTSSliceDataset(Dataset[dict[str, torch.Tensor]]):
    """Slice dataset that loads preprocessed cases from disk cache instead of re-processing.

    Uses .npz files pre-generated via preprocess_case(). Dramatically faster than
    re-running nibabel load + z-score + crop on every epoch.
    """

    def __init__(
        self,
        cases: list[BraTSCase],
        cache_dir: str | Path,
        include_empty: bool = False,
        augment: bool = False,
        cache_size: int = 8,
        target_shape: tuple[int, int] = (160, 160),
    ) -> None:
        self.cases = {case.case_id: case for case in cases}
        self.cache_dir = Path(cache_dir)
        validate_processed_cache(cases, self.cache_dir)
        self.records = build_slice_records_from_cache(cases, self.cache_dir, include_empty=include_empty)
        self.augmentor = SliceAugmentor(enabled=augment)
        self.mem_cache: OrderedDict[str, ProcessedCase] = OrderedDict()
        self.cache_size = cache_size
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.records)

    def _get_case(self, case_id: str) -> ProcessedCase:
        if case_id in self.mem_cache:
            cached = self.mem_cache.pop(case_id)
            self.mem_cache[case_id] = cached
            return cached
        processed = load_processed_cache(self.cache_dir, case_id)
        self.mem_cache[case_id] = processed
        while len(self.mem_cache) > self.cache_size:
            self.mem_cache.popitem(last=False)
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
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float(),
            "case_id": case_id,
            "slice_index": slice_index,
        }


class CachedRandomAugmentedSliceDataset(Dataset[dict[str, torch.Tensor]]):
    """Cached slice dataset with repeated on-the-fly random augmentation.

    For each original slice, __len__ exposes aug_samples_per_slice training draws,
    and each draw applies one random augmentation after loading the preprocessed cache.
    """

    def __init__(
        self,
        cases: list[BraTSCase],
        cache_dir: str | Path,
        include_empty: bool = False,
        aug_samples_per_slice: int = 1,
        noise_std: float = 0.05,
        target_shape: tuple[int, int] = (160, 160),
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cases = {case.case_id: case for case in cases}
        validate_processed_cache(cases, self.cache_dir)
        self.records = build_slice_records_from_cache(cases, self.cache_dir, include_empty=include_empty)
        self.aug_samples_per_slice = max(1, aug_samples_per_slice)
        self.augmentor = SliceAugmentor(enabled=True, noise_std=noise_std)
        self.mem_cache: OrderedDict[str, ProcessedCase] = OrderedDict()
        self.cache_size = 8
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.records) * self.aug_samples_per_slice

    def _get_case(self, case_id: str) -> ProcessedCase:
        if case_id in self.mem_cache:
            cached = self.mem_cache.pop(case_id)
            self.mem_cache[case_id] = cached
            return cached
        processed = load_processed_cache(self.cache_dir, case_id)
        self.mem_cache[case_id] = processed
        while len(self.mem_cache) > self.cache_size:
            self.mem_cache.popitem(last=False)
        return processed

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index // self.aug_samples_per_slice]
        processed = self._get_case(record["case_id"])
        image = processed["image"][:, record["slice_index"], :, :]
        target = processed["regions"][:, record["slice_index"], :, :]
        image = pad_or_crop_2d(image, self.target_shape)
        target = pad_or_crop_2d(target, self.target_shape)
        image, target = self.augmentor(image, target)
        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float(),
            "case_id": record["case_id"],
            "slice_index": record["slice_index"],
        }


class BraTSMultiSliceDataset(Dataset[dict[str, torch.Tensor]]):
    """2.5D dataset: stacks N consecutive 2D slices as multi-channel input.

    Each item returns N consecutive slices stacked along the channel dimension
    (N * 4 modalities = N*4 input channels). The target is the segmentation
    of the middle slice. Boundary slices are padded by replicating the edge slice.

    Args:
        cases: List of BraTSCase objects for this split.
        num_slices: Number of consecutive slices to stack (default 3).
        include_empty: If True, include slices with no tumor.
        augment: If True, apply data augmentation.
        cache_size: Number of preprocessed cases to keep in memory.
        target_shape: (H, W) to crop/pad slices to.
    """

    def __init__(
        self,
        cases: list[BraTSCase],
        num_slices: int = 3,
        include_empty: bool = False,
        augment: bool = False,
        cache_size: int = 2,
        target_shape: tuple[int, int] = (160, 160),
    ) -> None:
        self.num_slices = num_slices
        self.half = num_slices // 2
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

        total_slices = processed["image"].shape[1]
        image_stack = []
        for offset in range(-self.half, self.half + 1):
            idx = min(max(slice_index + offset, 0), total_slices - 1)
            slc = processed["image"][:, idx, :, :]
            slc = pad_or_crop_2d(slc, self.target_shape)
            image_stack.append(slc)

        target = processed["regions"][:, slice_index, :, :]
        target = pad_or_crop_2d(target, self.target_shape)

        stacked = np.concatenate(image_stack, axis=0)
        stacked, target = self.augmentor(stacked, target)

        return {
            "image": torch.from_numpy(stacked).float(),
            "target": torch.from_numpy(target).float(),
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
