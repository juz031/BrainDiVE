from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prfs import plot_prf_maps
from .utils import ROI_ORDER, atomic_write_dataframe, condition_map, load_rgb, output_root, stable_seed


def _primary(config: dict, feature: str) -> pd.DataFrame:
    frame = pd.read_csv(output_root(config) / "summaries" / f"voxel_summary_{feature}.csv")
    return frame[
        (frame["rank_cutoff"].astype(int) == int(config["images_per_voxel"])) &
        (frame["image_summary"] == config["statistics"]["primary_image_summary"])
    ].copy()


def _condition_style(config: dict):
    mapping = condition_map(config)
    order = [condition["id"] for condition in config["conditions"]]
    colors = [mapping[condition]["color"] for condition in order]
    labels = [mapping[condition]["display_name"] for condition in order]
    return mapping, order, colors, labels


def _finite_column_mean(values: np.ndarray) -> np.ndarray:
    """Column means without warnings when a resumable run has empty columns."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    totals = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(totals, counts, out=np.full(values.shape[1], np.nan), where=counts > 0)


def _finite_column_sem(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = _finite_column_mean(values)
    centered = np.where(finite, values - means, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0), counts - 1,
        out=np.full(values.shape[1], np.nan), where=counts > 1,
    )
    return np.divide(
        np.sqrt(variance), np.sqrt(counts),
        out=np.full(values.shape[1], np.nan), where=counts > 1,
    )


def scalar_comparison_figure(config: dict, feature_family: str, metric: str, backbone: str) -> Path:
    root = output_root(config)
    frame = _primary(config, feature_family)
    _, order, colors, labels = _condition_style(config)
    backbone_frame = frame[frame["backbone"] == backbone]
    scopes = [
        scope for scope in ("full", "prf")
        if backbone_frame.loc[backbone_frame["scope"] == scope, metric].notna().any()
    ]
    if not scopes:
        raise ValueError(f"No finite {feature_family}/{metric} values for {backbone}")
    fig, axes = plt.subplots(len(scopes), 4, figsize=(18, 4.4 * len(scopes)), sharey=True, squeeze=False)
    rng = np.random.default_rng(stable_seed("figure", feature_family, metric, backbone))
    for row_index, scope in enumerate(scopes):
        for column_index, roi in enumerate(ROI_ORDER):
            axis = axes[row_index, column_index]
            subset = frame[(frame["backbone"] == backbone) & (frame["roi"] == roi) & (frame["scope"] == scope)]
            for x, (condition, color) in enumerate(zip(order, colors)):
                values = subset.loc[subset["condition"] == condition, metric].dropna().to_numpy(float)
                jitter = rng.uniform(-0.14, 0.14, len(values))
                axis.scatter(x + jitter, values, s=25, alpha=0.65, color=color, edgecolor="white", linewidth=0.35)
                if len(values):
                    axis.hlines(np.median(values), x - 0.28, x + 0.28, color="black", linewidth=2.1)
                    low, high = np.percentile(values, [25, 75])
                    axis.vlines(x, low, high, color="black", linewidth=1.1)
            axis.set_title(roi, weight="bold")
            axis.grid(axis="y", color="#DDE2E5", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(range(len(order)), labels, rotation=35, ha="right", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(("Full image\n" if scope == "full" else "Generation-pRF weighted\n") + metric)
    fig.suptitle(
        f"{config['backbones'][backbone]['display_name']} · {metric} · all image conditions",
        fontsize=16, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = root / "figures" / "comparisons" / f"{backbone}__{feature_family}__{metric}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def gabor_profile_figure(config: dict, profile: str, backbone: str) -> Path:
    root = output_root(config)
    frame = _primary(config, "gabor")
    mapping, order, _, _ = _condition_style(config)
    if profile == "spatial_frequency":
        columns = sorted(column for column in frame if column.startswith("gabor_sf_profile_"))
        x = np.asarray(config["features"]["gabor"]["spatial_frequencies_cycles_per_image"], dtype=float)
        xlabel = "Spatial frequency (cycles/image)"
    else:
        columns = sorted(column for column in frame if column.startswith("gabor_orientation_profile_"))
        x = np.asarray(config["features"]["gabor"]["orientations_degrees"], dtype=float)
        xlabel = "Orientation (degrees)"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for row_index, scope in enumerate(("full", "prf")):
        for column_index, roi in enumerate(ROI_ORDER):
            axis = axes[row_index, column_index]
            subset = frame[(frame["backbone"] == backbone) & (frame["roi"] == roi) & (frame["scope"] == scope)]
            for condition in order:
                values = subset[subset["condition"] == condition][columns].to_numpy(float)
                if not len(values):
                    continue
                mean = _finite_column_mean(values)
                sem = _finite_column_sem(values)
                color = mapping[condition]["color"]
                axis.plot(x, mean, color=color, linewidth=2, label=mapping[condition]["display_name"])
                axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)
            if profile == "spatial_frequency":
                axis.set_xscale("log")
            axis.set_title(roi, weight="bold")
            axis.set_xlabel(xlabel)
            axis.grid(color="#E3E7EA", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                axis.set_ylabel("Full image\nGabor magnitude" if scope == "full" else "Generation-pRF weighted\nGabor magnitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{config['backbones'][backbone]['display_name']} · Gabor {profile.replace('_', ' ')} profile",
        fontsize=16, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = root / "figures" / "comparisons" / f"{backbone}__gabor__{profile}_profiles.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def gabor_heatmap_figure(config: dict, backbone: str, scope: str) -> Path:
    root = output_root(config)
    frame = _primary(config, "gabor")
    mapping, order, _, _ = _condition_style(config)
    n_sf = len(config["features"]["gabor"]["spatial_frequencies_cycles_per_image"])
    n_ori = len(config["features"]["gabor"]["orientations_degrees"])
    columns = [f"gabor_sf{sf:02d}_ori{ori:02d}" for sf in range(n_sf) for ori in range(n_ori)]
    subset_all = frame[(frame["backbone"] == backbone) & (frame["scope"] == scope)]
    arrays = []
    for roi in ROI_ORDER:
        for condition in order:
            values = subset_all[(subset_all["roi"] == roi) & (subset_all["condition"] == condition)][columns]
            arrays.append(_finite_column_mean(values.to_numpy(float)).reshape(n_sf, n_ori))
    finite_values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if not finite_values.size:
        raise ValueError(f"No finite Gabor values for {backbone}/{scope}")
    low = float(np.percentile(finite_values, 2))
    high = float(np.percentile(finite_values, 98))
    fig, axes = plt.subplots(4, len(order), figsize=(3.0 * len(order), 10.8), sharex=True, sharey=True)
    image = None
    for row_index, roi in enumerate(ROI_ORDER):
        for column_index, condition in enumerate(order):
            axis = axes[row_index, column_index]
            values = subset_all[(subset_all["roi"] == roi) & (subset_all["condition"] == condition)][columns]
            matrix = _finite_column_mean(values.to_numpy(float)).reshape(n_sf, n_ori)
            image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=low, vmax=high, origin="lower")
            if row_index == 0:
                axis.set_title(mapping[condition]["display_name"], fontsize=9, weight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{roi}\nfrequency index")
            if row_index == len(ROI_ORDER) - 1:
                axis.set_xlabel("orientation index")
    colorbar_axis = fig.add_axes([0.925, 0.19, 0.015, 0.63])
    fig.colorbar(image, cax=colorbar_axis, label="Mean Gabor magnitude across voxels")
    scope_label = "full image" if scope == "full" else "generation-pRF weighted"
    fig.suptitle(
        f"{config['backbones'][backbone]['display_name']} · {scope_label} · 8×12 Gabor maps · all conditions",
        fontsize=15, weight="bold",
    )
    fig.subplots_adjust(top=0.91, right=0.90, wspace=0.12, hspace=0.16)
    path = root / "figures" / "comparisons" / f"{backbone}__gabor__{scope}__heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def gallery_figure(config: dict, manifest: pd.DataFrame, prfs: pd.DataFrame, backbone: str, roi: str) -> Path:
    root = output_root(config)
    _, order, _, labels = _condition_style(config)
    contexts = manifest[(manifest["backbone"] == backbone) & (manifest["roi"] == roi)]
    voxel_rank = contexts["voxel_rank"].min()
    voxel_id = int(contexts.loc[contexts["voxel_rank"] == voxel_rank, "voxel_id"].iloc[0])
    selected = contexts[(contexts["voxel_id"] == voxel_id) & (contexts["image_rank"] == 1)]
    prf_row = prfs[(prfs["backbone"] == backbone) & (prfs["roi"] == roi) & (prfs["voxel_id"] == voxel_id)].iloc[0]
    mask = np.load(prf_row["mask_path"]).astype(float)
    display_mask = mask / max(mask.max(), 1e-12)
    fig, axes = plt.subplots(2, len(order), figsize=(3.1 * len(order), 6.4))
    for column, (condition, label) in enumerate(zip(order, labels)):
        row = selected[selected["condition"] == condition].iloc[0]
        image = load_rgb(row["image_path"], int(config["features"]["image_size"]))
        axes[0, column].imshow(image)
        axes[1, column].imshow((image.astype(float) * display_mask[..., None]).astype(np.uint8))
        axes[0, column].set_title(label, fontsize=9, weight="bold")
        for axis in axes[:, column]:
            axis.set_axis_off()
    axes[0, 0].set_ylabel("Full image", fontsize=11, weight="bold")
    axes[1, 0].set_ylabel("Generation-pRF display", fontsize=11, weight="bold")
    fig.suptitle(
        f"{config['backbones'][backbone]['display_name']} · {roi} · voxel {voxel_id} · rank-1 examples",
        fontsize=15, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = root / "figures" / "galleries" / f"{backbone}__{roi}__rank1_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def qc_figures(config: dict, manifest: pd.DataFrame) -> list[Path]:
    root = output_root(config)
    outputs = []
    counts = manifest.groupby(["backbone", "roi", "condition"]).size().reset_index(name="images")
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = counts.apply(lambda row: f"{row.backbone}\n{row.roi}\n{row.condition}", axis=1)
    ax.bar(np.arange(len(counts)), counts["images"], color="#567A9C")
    ax.set_xticks(np.arange(len(counts)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("Manifest images")
    ax.set_title("Input counts by backbone, ROI, and condition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = root / "figures" / "qc" / "manifest_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    overlap_path = root / "summaries" / "nsd_selection_overlap.csv"
    if overlap_path.is_file():
        overlap = pd.read_csv(overlap_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(overlap["intersection"], bins=np.arange(-0.5, 31.5, 1), color="#2A9D8F")
        ax.set_xlabel("Shared images between measured and predicted top-30 sets")
        ax.set_ylabel("Voxels")
        ax.set_title("NSD selection overlap")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = root / "figures" / "qc" / "nsd_selection_overlap.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_all_figures(config: dict) -> pd.DataFrame:
    root = output_root(config)
    manifest = pd.read_csv(root / "manifests" / "image_manifest.csv")
    prfs = pd.read_csv(root / "manifests" / "voxel_prf_manifest.csv")
    records = []
    for path in plot_prf_maps(config, prfs):
        records.append({"section": "pRF maps", "path": str(path), "caption": "Generation-model Gaussian pRFs on black canvases; display-normalized to peak 1."})
    basic_metrics = ["mean_L", "std_L", "mean_a", "std_a", "mean_b", "std_b", "mean_saturation", "std_saturation"]
    curvature_metrics = ["curvature", "edge_fraction", "edge_mass", "effective_edge_pixels"]
    for backbone in config["backbones"]:
        for metric in basic_metrics:
            path = scalar_comparison_figure(config, "basic", metric, backbone)
            records.append({"section": "Basic statistics", "path": str(path), "caption": "Dots are voxel medians across 30 images; horizontal bars are medians across voxels."})
        curvature_frame = _primary(config, "curvature")
        for metric in curvature_metrics:
            if metric not in curvature_frame:
                continue
            path = scalar_comparison_figure(config, "curvature", metric, backbone)
            records.append({"section": "Curvature", "path": str(path), "caption": "Curved-Gabor and edge summaries; panels show each metric's applicable full-image or generation-pRF scope."})
        for profile in ("spatial_frequency", "orientation"):
            path = gabor_profile_figure(config, profile, backbone)
            records.append({"section": "Gabor profiles", "path": str(path), "caption": "Lines average voxel summaries; ribbons are SEM across voxels."})
        for scope in ("full", "prf"):
            path = gabor_heatmap_figure(config, backbone, scope)
            records.append({"section": "Gabor heat maps", "path": str(path), "caption": "Each 8×12 map averages the 30-image voxel summaries across voxels."})
        for roi in ROI_ORDER:
            path = gallery_figure(config, manifest, prfs, backbone, roi)
            records.append({"section": "Image galleries", "path": str(path), "caption": "Rank-1 examples are illustrative and are not independent statistical observations."})
    for path in qc_figures(config, manifest):
        records.append({"section": "Quality control", "path": str(path), "caption": "Input and NSD-selection quality control."})
    index = pd.DataFrame(records)
    section_order = {
        section: order for order, section in enumerate((
            "pRF maps", "Basic statistics", "Curvature", "Gabor profiles",
            "Gabor heat maps", "Image galleries", "Quality control",
        ))
    }
    index["_section_order"] = index["section"].map(section_order)
    index = index.sort_values("_section_order", kind="stable").drop(columns="_section_order").reset_index(drop=True)
    atomic_write_dataframe(index, root / "figures" / "figure_index.csv")
    return index
