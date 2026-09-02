from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import atomic_write_dataframe, ensure_output_tree, output_root


def _prf_utils(config: dict):
    project_root = str(Path(config["paths"]["project_root"]))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import prf_utils
    return prf_utils


def generation_prf(config: dict, subject: str, backbone: str, voxel_id: int) -> tuple[int, np.ndarray, np.ndarray]:
    subject_dir = subject if subject.startswith("S") else f"S{subject}"
    model_dir = config["backbones"][backbone]["generation_model_dir"]
    path = Path(config["paths"]["model_root"]) / subject_dir / model_dir / "best_prf_idx.npy"
    indices = np.load(path, mmap_mode="r")
    if not 0 <= voxel_id < len(indices):
        raise IndexError(f"Voxel {voxel_id} is outside {path} with length {len(indices)}")
    prf_id = int(indices[voxel_id])
    prf_utils = _prf_utils(config)
    grid, _ = prf_utils.get_prf_grid(config["prf"]["grid"])
    if not 0 <= prf_id < len(grid):
        raise IndexError(f"pRF index {prf_id} is outside grid of length {len(grid)}")
    params = np.asarray(grid[prf_id], dtype=np.float64)
    mask = prf_utils.gauss_2d(
        center=params[:2].copy(), sd=float(params[2]),
        patch_size=int(config["prf"]["image_size"]),
    ).astype(np.float64)
    mask /= mask.sum()
    return prf_id, params, mask


def build_prf_manifest(config: dict, contexts: pd.DataFrame) -> pd.DataFrame:
    ensure_output_tree(config)
    root = output_root(config)
    rows = []
    unique = contexts[["subject", "backbone", "roi", "voxel_id", "voxel_rank"]].drop_duplicates()
    degrees = float(config["prf"]["display_degrees"])
    for record in unique.itertuples(index=False):
        prf_id, params, mask = generation_prf(
            config, record.subject, record.backbone, int(record.voxel_id)
        )
        x, y, sigma = map(float, params)
        mask_dir = root / "masks" / record.subject / record.backbone / record.roi
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"voxel-{int(record.voxel_id):06d}.npy"
        np.save(mask_path, mask.astype(np.float32))
        visible_peak = float(mask.max())
        effective_pixels = float(1.0 / np.sum(mask**2))
        rows.append({
            "subject": record.subject, "backbone": record.backbone, "roi": record.roi,
            "voxel_id": int(record.voxel_id), "voxel_rank": int(record.voxel_rank),
            "generation_prf_id": prf_id, "x_normalized": x, "y_normalized": y,
            "sigma_normalized": sigma, "x_degrees": x * degrees, "y_degrees": y * degrees,
            "sigma_degrees": sigma * degrees,
            "eccentricity_degrees": float(np.hypot(x, y) * degrees),
            "polar_angle_degrees": float(np.degrees(np.arctan2(y, x)) % 360),
            "visible_normalization": "sum_to_one_after_visible_image_crop",
            "visible_peak_weight": visible_peak, "effective_pixels": effective_pixels,
            "mask_path": str(mask_path),
            "checkpoint_path": str(Path(config["paths"]["model_root"]) / record.subject /
                                   config["backbones"][record.backbone]["generation_model_dir"] /
                                   "best_prf_idx.npy"),
        })
    frame = pd.DataFrame(rows).sort_values(["subject", "backbone", "roi", "voxel_rank"])
    atomic_write_dataframe(frame, root / "manifests" / "voxel_prf_manifest.csv")
    return frame


def plot_prf_maps(config: dict, prfs: pd.DataFrame) -> list[Path]:
    root = output_root(config)
    outputs: list[Path] = []
    size = int(config["prf"]["image_size"])
    for (subject, backbone, roi), group in prfs.groupby(["subject", "backbone", "roi"], sort=False):
        group = group.sort_values("voxel_rank")
        ncols = 5
        nrows = int(np.ceil(len(group) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 2.75 * nrows), facecolor="black")
        axes = np.asarray(axes).reshape(-1)
        for axis, row in zip(axes, group.itertuples(index=False)):
            mask = np.load(row.mask_path).astype(float)
            shown = mask / max(mask.max(), 1e-12)
            axis.imshow(shown, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            center_x = size / 2 + row.x_normalized * size
            center_y = size / 2 - row.y_normalized * size
            axis.scatter([center_x], [center_y], s=16, c="cyan", marker="+")
            axis.contour(shown, levels=[np.exp(-0.5)], colors=["white"], linewidths=0.8)
            axis.set_title(
                f"rank {row.voxel_rank:02d} · voxel {row.voxel_id}\n"
                f"x={row.x_degrees:.2f}°, y={row.y_degrees:.2f}°, σ={row.sigma_degrees:.2f}°",
                color="white", fontsize=8,
            )
            axis.set_axis_off()
            axis.set_facecolor("black")
        for axis in axes[len(group):]:
            axis.set_axis_off()
        fig.suptitle(
            f"{subject} · {config['backbones'][backbone]['display_name']} · {roi} · generation-model pRFs",
            color="white", fontsize=15, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = root / "figures" / "prf_maps" / f"{subject}__{backbone}__{roi}__estimated_prfs.png"
        fig.savefig(path, dpi=180, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs
