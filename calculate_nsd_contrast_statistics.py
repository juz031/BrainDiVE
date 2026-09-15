#!/usr/bin/env python3
"""Calculate per-image luminance and RMS contrast statistics for NSD images.

The reported contrast matches ``apply_rms_contrast_max`` in
``brain_guide_pipeline.py``: it is the population standard deviation of
Rec.709 luminance, without division by mean luminance.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_STIMULUS_DIR = Path("/lab_data/hendersonlab/datasets/nsd_preproc/stimuli")
DEFAULT_SPLIT_ROOT = Path("/user_data/junruz/prf_models/concat/split_1_zscore")
REC709_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-image Rec.709 mean luminance and RMS contrast for "
            "one NSD subject and summarize their distributions. Explicit file "
            "paths override subject-derived defaults; they are not checked "
            "for subject identity."
        )
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=range(1, 9),
        default=1,
        help="NSD subject number, used to select default paths (default: 1).",
    )
    parser.add_argument(
        "--stimulus-dir",
        type=Path,
        default=DEFAULT_STIMULUS_DIR,
        help="Directory containing S{subject}_stimuli_224.h5py files.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help="Directory containing S{subject} split folders and default JSON outputs.",
    )
    parser.add_argument(
        "--stimulus-file",
        type=Path,
        help="Override <stimulus-dir>/S{subject}_stimuli_224.h5py.",
    )
    parser.add_argument(
        "--dataset-key",
        default="stimuli",
        help="HDF5 dataset name or path (default: stimuli).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help=(
            "Override <split-root>/S{subject}/data_splits_S{subject}.pkl. Use "
            "--no-split to process every image."
        ),
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Ignore --split-file and process every image in the HDF5 dataset.",
    )
    parser.add_argument(
        "--split-id",
        type=int,
        default=1,
        help="BrainDiVE split index in the split pickle (default: 1).",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "nest", "val"),
        default=("train", "nest"),
        help="Split partitions to combine (default: train nest).",
    )
    parser.add_argument(
        "--indices-file",
        type=Path,
        help=(
            "Optional .npy file of image indices. This overrides split-file "
            "selection."
        ),
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=(5.0, 50.0, 95.0),
        help="Percentiles to report (default: 5 50 95).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images read from HDF5 at once (default: 64).",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        help=(
            "Pixel divisor used to map pixels to [0,1]. By default integer "
            "datasets use their dtype maximum and floating datasets use 1."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Override <split-root>/S{subject}/nsd_contrast_statistics.json.",
    )
    parser.add_argument(
        "--output-per-image",
        type=Path,
        default=None,
        help=(
            "Optional .npz path containing image_indices, luminance_mean, "
            "and rms_contrast in [0,1] units."
        ),
    )
    args = parser.parse_args(argv)
    subject_dir = args.split_root / f"S{args.subject}"
    if args.stimulus_file is None:
        args.stimulus_file = args.stimulus_dir / f"S{args.subject}_stimuli_224.h5py"
    if args.split_file is None:
        args.split_file = subject_dir / f"data_splits_S{args.subject}.pkl"
    if args.output_json is None:
        args.output_json = subject_dir / "nsd_contrast_statistics.json"
    return args


def load_split_indices(
    split_file: Path,
    split_id: int,
    partitions: list[str] | tuple[str, ...],
) -> np.ndarray:
    with open(split_file, "rb") as handle:
        splits = pickle.load(handle)

    try:
        selected_split = splits[split_id]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"Could not select split {split_id} from {split_file}."
        ) from error
    if not isinstance(selected_split, dict):
        raise TypeError(
            f"Split {split_id} must be a dictionary, got "
            f"{type(selected_split).__name__}."
        )

    missing = [name for name in partitions if name not in selected_split]
    if missing:
        raise KeyError(f"Missing partitions in split {split_id}: {missing}")
    indices = np.concatenate(
        [np.asarray(selected_split[name], dtype=np.int64) for name in partitions]
    )
    return np.unique(indices)


def select_indices(args: argparse.Namespace, image_count: int) -> np.ndarray:
    if args.indices_file is not None:
        indices = np.asarray(np.load(args.indices_file), dtype=np.int64).reshape(-1)
        source = str(args.indices_file)
    elif args.no_split:
        indices = np.arange(image_count, dtype=np.int64)
        source = "all images"
    else:
        indices = load_split_indices(
            args.split_file,
            args.split_id,
            args.partitions,
        )
        source = str(args.split_file)

    indices = np.unique(indices)
    if indices.size == 0:
        raise ValueError(f"Image selection from {source} is empty.")
    if indices[0] < 0 or indices[-1] >= image_count:
        raise IndexError(
            f"Image indices [{indices[0]}, {indices[-1]}] are outside a "
            f"dataset containing {image_count} images."
        )
    return indices


def images_to_bhwc(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected a four-dimensional image batch, got {images.shape}.")
    if images.shape[1] in (1, 3):
        return images.transpose(0, 2, 3, 1)
    if images.shape[-1] in (1, 3):
        return images
    raise ValueError(
        "Could not identify the color channel axis in image batch with shape "
        f"{images.shape}."
    )


def resolve_input_scale(dtype: np.dtype, requested_scale: float | None) -> float:
    if requested_scale is not None:
        if requested_scale <= 0:
            raise ValueError("--input-scale must be positive.")
        return float(requested_scale)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def batch_luminance_statistics(
    images: np.ndarray,
    input_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    images = images_to_bhwc(images).astype(np.float32) / input_scale
    if images.shape[-1] == 1:
        luminance = images[..., 0]
    else:
        luminance = np.einsum("bhwc,c->bhw", images, REC709_WEIGHTS)
    means = luminance.mean(axis=(-2, -1), dtype=np.float64)
    rms = np.sqrt(
        np.mean(
            (luminance - means[:, None, None]) ** 2,
            axis=(-2, -1),
            dtype=np.float64,
        )
    )
    return means, rms


def distribution_summary(values: np.ndarray, percentiles: np.ndarray) -> dict[str, Any]:
    percentile_values = np.percentile(values, percentiles)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            f"{percentile:g}": float(value)
            for percentile, value in zip(percentiles, percentile_values)
        },
    }


def scaled_summary(summary: dict[str, Any], scale: float) -> dict[str, Any]:
    return {
        key: (
            {percentile: value * scale for percentile, value in item.items()}
            if key == "percentiles"
            else item * scale
        )
        for key, item in summary.items()
    }


def calculate(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    percentiles = np.asarray(args.percentiles, dtype=np.float64)
    if np.any(~np.isfinite(percentiles)) or np.any((percentiles < 0) | (percentiles > 100)):
        raise ValueError("--percentiles values must be finite and between 0 and 100.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    with h5py.File(args.stimulus_file, "r") as handle:
        if args.dataset_key not in handle:
            raise KeyError(
                f"Dataset {args.dataset_key!r} not found in {args.stimulus_file}; "
                f"available top-level keys: {list(handle.keys())}"
            )
        dataset = handle[args.dataset_key]
        if dataset.ndim != 4:
            raise ValueError(
                f"Image dataset must have four dimensions, got {dataset.shape}."
            )
        indices = select_indices(args, dataset.shape[0])
        input_scale = resolve_input_scale(dataset.dtype, args.input_scale)

        mean_chunks = []
        rms_chunks = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            means, rms = batch_luminance_statistics(
                np.asarray(dataset[batch_indices]),
                input_scale,
            )
            mean_chunks.append(means)
            rms_chunks.append(rms)

        dataset_shape = list(dataset.shape)
        dataset_dtype = str(dataset.dtype)

    luminance_means = np.concatenate(mean_chunks)
    rms_contrasts = np.concatenate(rms_chunks)
    mean_unit = distribution_summary(luminance_means, percentiles)
    rms_unit = distribution_summary(rms_contrasts, percentiles)
    summary = {
        "subject": args.subject,
        "definition": (
            "Population standard deviation of Rec.709 luminance per image; "
            "not divided by mean luminance."
        ),
        "stimulus_file": str(args.stimulus_file),
        "dataset_key": args.dataset_key,
        "dataset_shape": dataset_shape,
        "dataset_dtype": dataset_dtype,
        "input_scale": input_scale,
        "image_count": int(len(indices)),
        "selection": (
            {"type": "indices_file", "path": str(args.indices_file)}
            if args.indices_file is not None
            else {"type": "all"}
            if args.no_split
            else {
                "type": "brain_dive_split",
                "split_file": str(args.split_file),
                "split_id": args.split_id,
                "partitions": list(args.partitions),
            }
        ),
        "luminance_mean": {
            "unit_interval": mean_unit,
            "byte_0_255": scaled_summary(mean_unit, 255.0),
        },
        "rms_contrast": {
            "unit_interval": rms_unit,
            "byte_0_255": scaled_summary(rms_unit, 255.0),
        },
    }
    return summary, indices, luminance_means, rms_contrasts


def main() -> None:
    args = parse_args()
    summary, indices, luminance_means, rms_contrasts = calculate(args)
    rendered = json.dumps(summary, indent=2)
    print(rendered)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    if args.output_per_image is not None:
        args.output_per_image.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_per_image,
            image_indices=indices,
            luminance_mean=luminance_means,
            rms_contrast=rms_contrasts,
        )


if __name__ == "__main__":
    main()
