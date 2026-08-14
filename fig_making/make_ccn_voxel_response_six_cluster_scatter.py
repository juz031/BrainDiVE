#!/usr/bin/env python3
"""Make a six-cluster response scatter plot for one BrainDiVE voxel.

The default example is V1 voxel 8066. Natural images come from the set-2
validation partition. Predictions use the frozen OpenCLIP ConvNeXt-Base set-2
encoding model and its own fitted pRF. The contrast-adjusted natural images use
the exact Rec.709 RMS matching function used during BrainDiVE MEI generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image

from brain_guide_pipeline import apply_rms_contrast_constraint
from make_ccn_top3_voxel_test_model_natural import test_model_predictions
from validation.heldout_model import (
    DEFAULT_ADV_CHECKPOINT,
    MODEL_SPECS,
    build_validation_encoder,
    load_validation_image_ids,
    load_validation_resources,
    preprocess,
)


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore")
DEFAULT_FEATURE_ROOT = Path("/user_data/junruz/prf_features")
DEFAULT_NSD_ROOT = Path("/lab_data/hendersonlab/datasets/nsd_preproc")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figure" / "ccn_voxel_response_six_cluster"

DEFAULT_DIFFUSION_RUN = Path(
    "/user_data/junruz/BrainDiVE/contrast_con_fixed/"
    "pRF_zscore_model_ranked/sub-01/"
    "gen-ADV_RN50-layer2__rank-DINO_RN50-layer2/"
    "split-01__steps-200__scale-1000__contrast-match-rms57.48"
    "__k-5__seeds-1000__keep-10"
)
DEFAULT_GRADIENT_RUN = Path(
    "/user_data/hanfeig/prf_max_activation/"
    "natural_mean_contrast_alignment_topr2_10k/S1/"
    "ADV_RN50_set1_gen__DINO_RN50_set1_rank__V1_layer2"
    "__ADV_layer2_topr2_voxels_009628_008066_010002_009712_007752"
    "__no_smooth__global_mean_lum116.21658776940036"
    "_rms57.48496223114593__10000seeds_keep10"
)

CLUSTERS = (
    ("natural_gt", "All natural\nNSD GT", "#4C78A8"),
    ("natural_pred", "All natural\nTest prediction", "#59A14F"),
    ("natural_top10", "Top 10 natural\nTest prediction", "#8CD17D"),
    (
        "natural_top10_contrast",
        "Top 10 natural\ncontrast matched",
        "#F2CF5B",
    ),
    ("diffusion_top10", "Top 10 diffusion\nTest prediction", "#F28E2B"),
    ("gradient_top10", "Top 10 gradient\nTest prediction", "#E15759"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-id", type=int, default=8066)
    parser.add_argument("--roi", default="V1")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--natural-set", type=int, default=2)
    parser.add_argument("--test-model", default="OPEN_CLIP_CONVNEXT_BASE")
    parser.add_argument("--test-set", type=int, default=2)
    parser.add_argument("--test-layer", default="stage2")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument(
        "--contrast-stats",
        type=Path,
        default=(
            DEFAULT_MODEL_ROOT / "S1" / "nsd_contrast_statistics.json"
        ),
    )
    parser.add_argument(
        "--diffusion-run",
        type=Path,
        default=DEFAULT_DIFFUSION_RUN,
    )
    parser.add_argument(
        "--gradient-run",
        type=Path,
        default=DEFAULT_GRADIENT_RUN,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_font() -> str:
    for font_path in Path("/usr/share/fonts/urw-base35").glob(
        "NimbusSans-*.otf"
    ):
        font_manager.fontManager.addfont(font_path)
    available = {item.name for item in font_manager.fontManager.ttflist}
    chosen = "Arial" if "Arial" in available else "Nimbus Sans"
    mpl.rcParams.update(
        {
            "font.family": chosen,
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
        }
    )
    return chosen


def load_natural_data(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_ids = np.sort(
        load_validation_image_ids(
            model_root=args.model_root,
            subject=args.subject,
            validation_set=args.natural_set,
        )
    )
    all_predictions = test_model_predictions(
        model_root=args.model_root,
        feature_root=args.feature_root,
        test_model=args.test_model,
        test_set=args.test_set,
        test_layer=args.test_layer,
        voxel_id=args.voxel_id,
    )
    predictions = np.asarray(all_predictions[val_ids], dtype=np.float64)

    beta_path = (
        args.nsd_root / "data" / f"S{args.subject}_betas_avg_bigmask.hdf5"
    )
    with h5py.File(beta_path, "r") as handle:
        observed = np.asarray(
            handle["betas"][val_ids, args.voxel_id],
            dtype=np.float64,
        )

    finite = np.isfinite(predictions)
    if finite.sum() < 10:
        raise ValueError("Fewer than 10 finite natural-image predictions")
    finite_indices = np.flatnonzero(finite)
    top_local = finite_indices[
        np.argsort(predictions[finite_indices], kind="stable")[-10:][::-1]
    ]
    top_ids = val_ids[top_local]

    stimulus_path = (
        args.nsd_root / "stimuli" / f"S{args.subject}_stimuli_224.h5py"
    )
    with h5py.File(stimulus_path, "r") as handle:
        arrays = np.stack(
            [
                np.asarray(handle["stimuli"][int(image_id)], dtype=np.uint8)
                for image_id in top_ids
            ]
        )
    if arrays.shape[1] == 3:
        arrays = np.moveaxis(arrays, 1, -1)
    return val_ids, observed, predictions, arrays


def load_target_contrast(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["rms_contrast"]["byte_0_255"]["mean"])


def contrast_match(images: np.ndarray, target_rms_byte: float) -> np.ndarray:
    tensor = (
        torch.from_numpy(np.asarray(images, dtype=np.float32))
        .permute(0, 3, 1, 2)
        / 255.0
    )
    matched = apply_rms_contrast_constraint(
        tensor,
        method="match",
        target_contrast=target_rms_byte / 255.0,
    )
    return matched


def diffusion_paths(args: argparse.Namespace) -> list[Path]:
    matches = sorted(
        args.diffusion_run.glob(
            f"roi-{args.roi}/voxel-rank-*__id-{args.voxel_id:06d}/ranking.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one diffusion ranking for voxel {args.voxel_id}, "
            f"found {len(matches)} under {args.diffusion_run}"
        )
    with matches[0].open("r", encoding="utf-8") as handle:
        ranking = json.load(handle)
    paths = []
    for result in ranking["results"]:
        candidate = Path(result["image_path"])
        if not candidate.is_file() and result.get("image_file"):
            candidate = matches[0].parent / result["image_file"]
        if candidate.is_file():
            paths.append(candidate)
    if len(paths) < 10:
        raise ValueError(f"Only {len(paths)} diffusion candidates were found")
    return paths


def gradient_paths(args: argparse.Namespace) -> list[Path]:
    csv_path = args.gradient_run / "all_ranking_candidates.csv"
    paths = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["voxel_idx"]) == args.voxel_id
                and row["roi"] == args.roi
                and row.get("condition", "contrast_only") == "contrast_only"
            ):
                candidate = Path(row["image_path"])
                if candidate.is_file():
                    paths.append(candidate)
    if len(paths) < 10:
        raise ValueError(f"Only {len(paths)} gradient candidates were found")
    return paths


def paths_to_tensor(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(
                image.convert("RGB").resize(
                    (224, 224),
                    Image.Resampling.BICUBIC,
                ),
                dtype=np.float32,
            ).copy()
        tensors.append(torch.from_numpy(pixels).permute(2, 0, 1) / 255.0)
    return torch.stack(tensors)


def score_tensors(
    *,
    images: torch.Tensor,
    voxel_id: int,
    resources,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(resources.best_weights[str(voxel_id)]).reshape(-1)
    weights = torch.as_tensor(values[:-1], dtype=torch.float64, device=device)
    intercept = torch.as_tensor(values[-1], dtype=torch.float64, device=device)
    mean = torch.as_tensor(
        resources.feature_mean[voxel_id],
        dtype=torch.float64,
        device=device,
    )
    std = torch.as_tensor(
        resources.feature_std[voxel_id],
        dtype=torch.float64,
        device=device,
    )
    prf_idx = int(resources.best_prf_idx[voxel_id])
    kernel = torch.as_tensor(
        resources.prf_stack[:, :, prf_idx],
        dtype=torch.float64,
        device=device,
    )

    scores = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            features = resources.encoder(
                preprocess(batch, resources.model)
            )[resources.layer]
            pooled = torch.einsum(
                "bchw,hw->bc",
                features.to(torch.float64),
                kernel,
            )
            prediction = ((pooled - mean) / std) @ weights + intercept
            scores.append(prediction.cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def make_plot(
    groups: dict[str, np.ndarray],
    args: argparse.Namespace,
    font_name: str,
    prf_idx: int,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(8066)
    fig, ax = plt.subplots(figsize=(10.6, 6.1), facecolor="white")

    labels = []
    for x, (key, label, color) in enumerate(CLUSTERS):
        values = np.asarray(groups[key], dtype=np.float64)
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-0.19, 0.19, len(values))
        large_group = len(values) > 50
        ax.scatter(
            x + jitter,
            values,
            s=20 if large_group else 54,
            color=color,
            alpha=0.32 if large_group else 0.86,
            edgecolor="none",
            rasterized=False,
            zorder=2,
        )
        mean = float(np.mean(values))
        ax.hlines(
            mean,
            x - 0.30,
            x + 0.30,
            colors=color,
            linewidth=3.2,
            zorder=3,
        )
        labels.append(label)

    ax.set_xticks(np.arange(len(CLUSTERS)))
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Response")
    ax.set_title(
        f"{args.roi} voxel {args.voxel_id} | "
        f"test model: OpenCLIP ConvNeXt-Base set {args.test_set} "
        f"(pRF {prf_idx})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_linewidth(1.1)
    ax.tick_params(width=1.1, length=5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.55, zorder=0)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.29)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.roi}_voxel-{args.voxel_id:06d}"
        f"__six-cluster-response-scatter"
    )
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def save_data(
    groups: dict[str, np.ndarray],
    identifiers: dict[str, list[str]],
    args: argparse.Namespace,
    target_rms: float,
    prf_idx: int,
    font_name: str,
) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / (
        f"{args.roi}_voxel-{args.voxel_id:06d}__scatter-points.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cluster", "response", "image_id_or_path"),
        )
        writer.writeheader()
        for key, _, _ in CLUSTERS:
            for value, identifier in zip(groups[key], identifiers[key]):
                writer.writerow(
                    {
                        "cluster": key,
                        "response": float(value),
                        "image_id_or_path": identifier,
                    }
                )

    summary_path = args.output_dir / (
        f"{args.roi}_voxel-{args.voxel_id:06d}__scatter-summary.json"
    )
    summary = {
        "voxel_id": args.voxel_id,
        "roi": args.roi,
        "natural_split": args.natural_set,
        "natural_partition": "val",
        "test_model": args.test_model,
        "test_set": args.test_set,
        "test_layer": args.test_layer,
        "test_model_prf_idx": prf_idx,
        "contrast_method": "BrainDiVE Rec.709 iterative RMS match",
        "target_mean_rms_contrast_byte": target_rms,
        "font": font_name,
        "clusters": {
            key: {
                "n": int(len(groups[key])),
                "mean": float(np.mean(groups[key])),
                "median": float(np.median(groups[key])),
                "min": float(np.min(groups[key])),
                "max": float(np.max(groups[key])),
            }
            for key, _, _ in CLUSTERS
        },
        "diffusion_run": str(args.diffusion_run),
        "gradient_run": str(args.gradient_run),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


def main() -> None:
    args = parse_args()
    font_name = configure_font()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    val_ids, observed, natural_pred, top_natural_images = load_natural_data(args)
    finite = np.flatnonzero(np.isfinite(natural_pred))
    top_local = finite[
        np.argsort(natural_pred[finite], kind="stable")[-10:][::-1]
    ]
    top_ids = val_ids[top_local]
    target_rms = load_target_contrast(args.contrast_stats)
    matched_top = contrast_match(top_natural_images, target_rms)

    diff_paths = diffusion_paths(args)
    grad_paths = gradient_paths(args)
    all_candidate_tensors = torch.cat(
        (
            matched_top,
            paths_to_tensor(diff_paths),
            paths_to_tensor(grad_paths),
        ),
        dim=0,
    )

    model = MODEL_SPECS[args.test_model]
    encoder = build_validation_encoder(
        model,
        device,
        DEFAULT_ADV_CHECKPOINT,
    )
    resources = load_validation_resources(
        layer=args.test_layer,
        model=model,
        subject=args.subject,
        validation_set=args.test_set,
        model_root=args.model_root,
        encoder=encoder,
    )
    candidate_scores = score_tensors(
        images=all_candidate_tensors,
        voxel_id=args.voxel_id,
        resources=resources,
        device=device,
        batch_size=args.batch_size,
    )
    matched_scores = candidate_scores[:10]
    diffusion_scores = candidate_scores[10 : 10 + len(diff_paths)]
    gradient_scores = candidate_scores[10 + len(diff_paths) :]
    diff_top = np.argsort(diffusion_scores, kind="stable")[-10:][::-1]
    grad_top = np.argsort(gradient_scores, kind="stable")[-10:][::-1]

    groups = {
        "natural_gt": observed,
        "natural_pred": natural_pred,
        "natural_top10": natural_pred[top_local],
        "natural_top10_contrast": matched_scores,
        "diffusion_top10": diffusion_scores[diff_top],
        "gradient_top10": gradient_scores[grad_top],
    }
    identifiers = {
        "natural_gt": [str(value) for value in val_ids],
        "natural_pred": [str(value) for value in val_ids],
        "natural_top10": [str(value) for value in top_ids],
        "natural_top10_contrast": [str(value) for value in top_ids],
        "diffusion_top10": [str(diff_paths[index]) for index in diff_top],
        "gradient_top10": [str(grad_paths[index]) for index in grad_top],
    }

    png_path, pdf_path = make_plot(
        groups,
        args,
        font_name,
        int(resources.best_prf_idx[args.voxel_id]),
    )
    csv_path, summary_path = save_data(
        groups,
        identifiers,
        args,
        target_rms,
        int(resources.best_prf_idx[args.voxel_id]),
        font_name,
    )
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
