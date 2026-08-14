#!/usr/bin/env python3
"""Create tight eight-cluster response scatters for shared ADV/DINO voxels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from make_ccn_top3_voxel_test_model_natural import test_model_predictions
from make_ccn_voxel_response_six_cluster_scatter import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_NSD_ROOT,
    configure_font,
    contrast_match,
    load_target_contrast,
    paths_to_tensor,
    score_tensors,
)
from validation.heldout_model import (
    DEFAULT_ADV_CHECKPOINT,
    MODEL_SPECS,
    build_validation_encoder,
    load_validation_image_ids,
    load_validation_resources,
)


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "figure" / "ccn_selected_voxel_response_eight_cluster"
)
SHARED_VOXELS = {
    "V1": (9656, 8066, 7752),
    "V4": (13177, 13162, 13641),
}
ROI_CONFIG = {
    "V1": {"source_roi": "V1", "layer": "layer2", "test_layer": "stage2"},
    "V4": {"source_roi": "hV4", "layer": "layer3", "test_layer": "stage3"},
}

DIFFUSION_BASE = Path(
    "/user_data/junruz/BrainDiVE/contrast_con_fixed/"
    "pRF_zscore_model_ranked/sub-01"
)
GRADIENT_SHARE_ROOT = Path(
    "/user_data/junruz/BrainDiVE/model_pair_top10_individual_rf_share"
)
TOP_IMAGE_COUNT = 8

CLUSTERS = (
    ("natural_gt", "Nat. NSD GT", "#4C78A8"),
    ("natural_pred", "Nat. pred.", "#59A14F"),
    ("natural_top", "Nat. Top 8 pred.", "#8CD17D"),
    (
        "natural_contrast",
        "Nat. Top 8 pred CM.",
        "#EDC948",
    ),
    ("diffusion_adv", "ADV diffusion", "#F28E2B"),
    ("diffusion_dino", "DINO diffusion", "#B279A2"),
    ("gradient_adv", "ADV gradient", "#E15759"),
    ("gradient_dino", "DINO gradient", "#9C755F"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roi",
        choices=("all", "V1", "V4"),
        default="all",
    )
    parser.add_argument(
        "--voxel-id",
        type=int,
        action="append",
        default=None,
        help="Optional voxel ID; repeat to select multiple voxels.",
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--natural-set", type=int, default=2)
    parser.add_argument("--test-model", default="OPEN_CLIP_CONVNEXT_BASE")
    parser.add_argument("--test-set", type=int, default=2)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--plot-from-csv",
        action="store_true",
        help="Rebuild the combined plot from existing point-level CSV files.",
    )
    return parser.parse_args()


def selected_voxels(args: argparse.Namespace, roi: str) -> tuple[int, ...]:
    if args.voxel_id is None:
        return SHARED_VOXELS[roi]
    allowed = set(SHARED_VOXELS[roi])
    chosen = tuple(value for value in args.voxel_id if value in allowed)
    return chosen


def diffusion_run(roi: str, generator: str) -> Path:
    config = ROI_CONFIG[roi]
    ranker = "DINO" if generator == "ADV" else "ADV"
    layer = config["layer"]
    return (
        DIFFUSION_BASE
        / f"gen-{generator}_RN50-{layer}__rank-{ranker}_RN50-{layer}"
        / (
            "split-01__steps-200__scale-1000__contrast-match-rms57.48"
            "__k-5__seeds-1000__keep-10"
        )
    )


def diffusion_paths(roi: str, voxel_id: int, generator: str) -> list[Path]:
    root = diffusion_run(roi, generator)
    source_roi = ROI_CONFIG[roi]["source_roi"]
    matches = sorted(
        root.glob(
            f"roi-{source_roi}/voxel-rank-*__id-{voxel_id:06d}/ranking.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {generator} diffusion ranking for {roi} "
            f"voxel {voxel_id}; found {len(matches)}"
        )
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    paths = []
    for result in payload["results"]:
        path = Path(result["image_path"])
        if not path.is_file() and result.get("image_file"):
            path = matches[0].parent / result["image_file"]
        if path.is_file():
            paths.append(path)
    if len(paths) < 10:
        raise ValueError(
            f"Only {len(paths)} {generator} diffusion images for {voxel_id}"
        )
    return paths


def gradient_paths(roi: str, voxel_id: int, generator: str) -> list[Path]:
    ranker = "DINO" if generator == "ADV" else "ADV"
    pair_dir = (
        GRADIENT_SHARE_ROOT
        / f"{generator}_set1_to_{ranker}_set1"
        / "gradient_hanfei"
        / f"{roi}_{ROI_CONFIG[roi]['layer']}"
    )
    matches = sorted(pair_dir.glob(f"voxel-rank-*_id-{voxel_id:06d}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {generator} gradient directory for {roi} "
            f"voxel {voxel_id}; found {len(matches)}"
        )
    paths = sorted(matches[0].glob("rank-*__original.png"))
    if len(paths) < TOP_IMAGE_COUNT:
        raise ValueError(
            f"Only {len(paths)} {generator} gradient images for "
            f"{roi} voxel {voxel_id}"
        )
    return paths


def exclude_all_black(paths: list[Path]) -> list[Path]:
    """Remove failed MEIs whose RGB pixels are all exactly zero."""
    retained = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        if int(np.max(pixels)) > 0:
            retained.append(path)
    return retained


def natural_data(
    args: argparse.Namespace,
    voxel_id: int,
    test_layer: str,
    validation_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_predictions = test_model_predictions(
        model_root=args.model_root,
        feature_root=args.feature_root,
        test_model=args.test_model,
        test_set=args.test_set,
        test_layer=test_layer,
        voxel_id=voxel_id,
    )
    predictions = np.asarray(all_predictions[validation_ids], dtype=np.float64)
    beta_path = (
        args.nsd_root / "data" / f"S{args.subject}_betas_avg_bigmask.hdf5"
    )
    with h5py.File(beta_path, "r") as handle:
        observed = np.asarray(
            handle["betas"][validation_ids, voxel_id],
            dtype=np.float64,
        )
    finite = np.flatnonzero(np.isfinite(predictions))
    top_local = finite[
        np.argsort(predictions[finite], kind="stable")[
            -TOP_IMAGE_COUNT:
        ][::-1]
    ]
    top_ids = validation_ids[top_local]
    stimulus_path = (
        args.nsd_root / "stimuli" / f"S{args.subject}_stimuli_224.h5py"
    )
    with h5py.File(stimulus_path, "r") as handle:
        images = np.stack(
            [
                np.asarray(handle["stimuli"][int(image_id)], dtype=np.uint8)
                for image_id in top_ids
            ]
        )
    if images.shape[1] == 3:
        images = np.moveaxis(images, 1, -1)
    return observed, predictions, top_local, images


def rank_top(
    scores: np.ndarray,
    paths: list[Path],
    count: int = 10,
) -> tuple[np.ndarray, list[Path]]:
    finite = np.flatnonzero(np.isfinite(scores))
    order = finite[
        np.argsort(scores[finite], kind="stable")[-count:][::-1]
    ]
    return scores[order], [paths[index] for index in order]


def plot_voxel(
    *,
    groups: dict[str, np.ndarray],
    roi: str,
    voxel_id: int,
    prf_idx: int,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(voxel_id)
    spacing = 0.70
    positions = np.arange(len(CLUSTERS), dtype=np.float64) * spacing
    fig, ax = plt.subplots(figsize=(8.8, 5.15), facecolor="white")
    for x, (key, _, color) in zip(positions, CLUSTERS):
        values = np.asarray(groups[key], dtype=np.float64)
        values = values[np.isfinite(values)]
        large = len(values) > 50
        jitter = rng.uniform(-0.105, 0.105, len(values))
        ax.scatter(
            x + jitter,
            values,
            s=16 if large else 46,
            color=color,
            alpha=0.28 if large else 0.86,
            edgecolor="none",
            zorder=2,
        )
        ax.hlines(
            float(np.mean(values)),
            x - 0.17,
            x + 0.17,
            color=color,
            linewidth=3.0,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [label for _, label, _ in CLUSTERS],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Response")
    ax.set_title(f"{roi} voxel {voxel_id}")
    ax.set_xlim(positions[0] - 0.30, positions[-1] + 0.30)
    ax.margins(x=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4, pad=2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.52, zorder=0)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.905, bottom=0.265)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{roi}_voxel-{voxel_id:06d}__eight-cluster-response-scatter"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_combined(
    *,
    voxel_panels: list[tuple[str, int, dict[str, np.ndarray]]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    short_labels = (
        "Nat. NSD GT",
        "Nat. pred.",
        "Nat. Top 8 pred.",
        "Nat. Top 8 pred CM.",
        "ADV diff.",
        "DINO diff.",
        "ADV grad.",
        "DINO grad.",
    )
    spacing = 0.58
    positions = np.arange(len(CLUSTERS), dtype=np.float64) * spacing
    fig, axes = plt.subplots(
        1,
        len(voxel_panels),
        figsize=(30.0, 5.9),
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    for panel_index, (roi, voxel_id, groups) in enumerate(voxel_panels):
        ax = axes[0, panel_index]
        rng = np.random.default_rng(voxel_id)
        for x, (key, _, color) in zip(positions, CLUSTERS):
            values = np.asarray(groups[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            large = len(values) > 50
            jitter = rng.uniform(-0.082, 0.082, len(values))
            ax.scatter(
                x + jitter,
                values,
                s=9 if large else 28,
                color=color,
                alpha=0.25 if large else 0.84,
                edgecolor="none",
                zorder=2,
            )
            ax.hlines(
                float(np.mean(values)),
                x - 0.13,
                x + 0.13,
                color=color,
                linewidth=2.3,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            short_labels,
            rotation=48,
            ha="right",
            rotation_mode="anchor",
            fontsize=14.5,
        )
        ax.set_title(f"{roi} voxel {voxel_id}", fontsize=21, pad=8)
        ax.set_xlim(positions[0] - 0.20, positions[-1] + 0.20)
        ax.margins(x=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(panel_index == 0)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(
            axis="x",
            width=0.9,
            length=3.5,
            pad=1,
        )
        ax.tick_params(
            axis="y",
            left=panel_index == 0,
            labelleft=panel_index == 0,
            width=0.9,
            length=4,
            labelsize=16,
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.5, zorder=0)
        if panel_index == 0:
            target_values = np.asarray(
                groups["diffusion_adv"],
                dtype=np.float64,
            )
            target_y = float(np.nanmin(target_values))
            target_x = float(positions[4])
            ax.annotate(
                "One image",
                xy=(target_x, target_y),
                xytext=(positions[3] - 0.12, target_y - 0.92),
                ha="center",
                va="center",
                fontsize=14.5,
                color="black",
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#666666",
                    "linewidth": 1.5,
                    "connectionstyle": "arc3,rad=-0.22",
                    "shrinkA": 3,
                    "shrinkB": 4,
                },
                zorder=5,
            )
    axes[0, 0].set_ylabel("Response", fontsize=23)
    fig.subplots_adjust(
        left=0.035,
        right=0.997,
        top=0.89,
        bottom=0.30,
        wspace=0.055,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "ADV_DINO_shared-voxels__eight-cluster-response-scatter__1x6"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        pdf_path,
        format="pdf",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    return png_path, pdf_path


def save_voxel_data(
    *,
    groups: dict[str, np.ndarray],
    identifiers: dict[str, list[str]],
    roi: str,
    voxel_id: int,
    prf_idx: int,
    output_dir: Path,
    target_rms: float,
) -> tuple[Path, Path]:
    csv_path = output_dir / (
        f"{roi}_voxel-{voxel_id:06d}__eight-cluster-points.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cluster", "response", "image_id_or_path"),
        )
        writer.writeheader()
        for key, _, _ in CLUSTERS:
            for response, identifier in zip(groups[key], identifiers[key]):
                writer.writerow(
                    {
                        "cluster": key,
                        "response": float(response),
                        "image_id_or_path": identifier,
                    }
                )
    summary = {
        "roi": roi,
        "voxel_id": voxel_id,
        "test_model": "OPEN_CLIP_CONVNEXT_BASE_set2",
        "test_model_prf_idx": prf_idx,
        "natural_split": 2,
        "natural_partition": "val",
        "target_mean_rms_contrast_byte": target_rms,
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
    }
    json_path = output_dir / (
        f"{roi}_voxel-{voxel_id:06d}__eight-cluster-summary.json"
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    configure_font()
    if args.plot_from_csv:
        requested_rois = ("V1", "V4") if args.roi == "all" else (args.roi,)
        voxel_panels = []
        for roi in requested_rois:
            for voxel_id in selected_voxels(args, roi):
                csv_path = args.output_dir / (
                    f"{roi}_voxel-{voxel_id:06d}__eight-cluster-points.csv"
                )
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                groups = {
                    key: np.asarray(
                        [
                            float(row["response"])
                            for row in rows
                            if row["cluster"] == key
                        ],
                        dtype=np.float64,
                    )
                    for key, _, _ in CLUSTERS
                }
                voxel_panels.append((roi, voxel_id, groups))
        png_path, pdf_path = plot_combined(
            voxel_panels=voxel_panels,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        print(png_path)
        print(pdf_path)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    validation_ids = np.sort(
        load_validation_image_ids(
            model_root=args.model_root,
            subject=args.subject,
            validation_set=args.natural_set,
        )
    )
    target_rms = load_target_contrast(args.contrast_stats)
    requested_rois = ("V1", "V4") if args.roi == "all" else (args.roi,)
    model = MODEL_SPECS[args.test_model]
    encoder = build_validation_encoder(model, device, DEFAULT_ADV_CHECKPOINT)

    outputs = []
    voxel_panels: list[tuple[str, int, dict[str, np.ndarray]]] = []
    for roi in requested_rois:
        test_layer = ROI_CONFIG[roi]["test_layer"]
        resources = load_validation_resources(
            layer=test_layer,
            model=model,
            subject=args.subject,
            validation_set=args.test_set,
            model_root=args.model_root,
            encoder=encoder,
        )
        for voxel_id in selected_voxels(args, roi):
            observed, predictions, top_local, top_images = natural_data(
                args,
                voxel_id,
                test_layer,
                validation_ids,
            )
            top_ids = validation_ids[top_local]
            adjusted = contrast_match(top_images, target_rms)
            adv_diff = exclude_all_black(
                diffusion_paths(roi, voxel_id, "ADV")
            )
            dino_diff = exclude_all_black(
                diffusion_paths(roi, voxel_id, "DINO")
            )
            adv_grad = exclude_all_black(
                gradient_paths(roi, voxel_id, "ADV")
            )
            dino_grad = exclude_all_black(
                gradient_paths(roi, voxel_id, "DINO")
            )

            tensors = (
                adjusted,
                paths_to_tensor(adv_diff),
                paths_to_tensor(dino_diff),
                paths_to_tensor(adv_grad),
                paths_to_tensor(dino_grad),
            )
            lengths = [len(tensor) for tensor in tensors]
            scores = score_tensors(
                images=torch.cat(tensors, dim=0),
                voxel_id=voxel_id,
                resources=resources,
                device=device,
                batch_size=args.batch_size,
            )
            offsets = np.cumsum((0, *lengths))
            adjusted_scores = scores[offsets[0] : offsets[1]]
            adv_diff_scores, adv_diff_paths = rank_top(
                scores[offsets[1] : offsets[2]],
                adv_diff,
                count=TOP_IMAGE_COUNT,
            )
            dino_diff_scores, dino_diff_paths = rank_top(
                scores[offsets[2] : offsets[3]],
                dino_diff,
                count=TOP_IMAGE_COUNT,
            )
            adv_grad_scores, adv_grad_paths = rank_top(
                scores[offsets[3] : offsets[4]],
                adv_grad,
                count=TOP_IMAGE_COUNT,
            )
            dino_grad_scores, dino_grad_paths = rank_top(
                scores[offsets[4] : offsets[5]],
                dino_grad,
                count=TOP_IMAGE_COUNT,
            )
            groups = {
                "natural_gt": observed,
                "natural_pred": predictions,
                "natural_top": predictions[top_local],
                "natural_contrast": adjusted_scores,
                "diffusion_adv": adv_diff_scores,
                "diffusion_dino": dino_diff_scores,
                "gradient_adv": adv_grad_scores,
                "gradient_dino": dino_grad_scores,
            }
            voxel_panels.append((roi, voxel_id, groups))
            identifiers = {
                "natural_gt": [str(value) for value in validation_ids],
                "natural_pred": [str(value) for value in validation_ids],
                "natural_top": [str(value) for value in top_ids],
                "natural_contrast": [str(value) for value in top_ids],
                "diffusion_adv": [str(value) for value in adv_diff_paths],
                "diffusion_dino": [str(value) for value in dino_diff_paths],
                "gradient_adv": [str(value) for value in adv_grad_paths],
                "gradient_dino": [str(value) for value in dino_grad_paths],
            }
            prf_idx = int(resources.best_prf_idx[voxel_id])
            png_path, pdf_path = plot_voxel(
                groups=groups,
                roi=roi,
                voxel_id=voxel_id,
                prf_idx=prf_idx,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
            csv_path, json_path = save_voxel_data(
                groups=groups,
                identifiers=identifiers,
                roi=roi,
                voxel_id=voxel_id,
                prf_idx=prf_idx,
                output_dir=args.output_dir,
                target_rms=target_rms,
            )
            outputs.extend((png_path, pdf_path, csv_path, json_path))
            print(png_path, flush=True)

    if voxel_panels:
        combined_png, combined_pdf = plot_combined(
            voxel_panels=voxel_panels,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        outputs.extend((combined_png, combined_pdf))
        print(combined_png, flush=True)
    print(f"Created {len(outputs)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
