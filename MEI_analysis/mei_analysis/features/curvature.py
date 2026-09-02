from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.filters import roberts

from ..utils import load_gray
from .common import identity_frame


def _legacy_bent(config: dict):
    legacy = str(Path(config["paths"]["legacy_image_stats"]))
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    import bent_gabor_model
    return bent_gabor_model


def build_extractor(config: dict, device: torch.device):
    settings = config["features"]["curvature"]
    size = int(config["features"]["image_size"])
    model = _legacy_bent(config)
    bank = model.bent_filter_bank(
        orients_deg=np.asarray(settings["orientations_degrees"], dtype=float),
        freqs_cpp=np.asarray([float(settings["spatial_frequency_cycles_per_image"]) / size]),
        bend_values=np.asarray(settings["bends"], dtype=float),
        spat_freq_bw=float(settings["spatial_frequency_bandwidth"]),
        spat_aspect_ratio=float(settings["spatial_aspect_ratio"]),
        n_sd_out=int(settings["n_sd_out"]), image_size=size,
        padding_mode=settings["padding_mode"], mode="curved",
    )
    extractor = model.bent_gabor_feature_extractor_spat(bank, device=device)
    _, _, _, bend_labels = bank.get_filters_spat()
    _, integer_labels = np.unique(bend_labels, return_inverse=True)
    return extractor, np.squeeze(integer_labels)


def extract_curvature(
    config: dict, rows: pd.DataFrame, prf: np.ndarray,
    device: torch.device, batch_size: int = 1,
) -> pd.DataFrame:
    extractor, bend_integer_labels = build_extractor(config, device)
    size = int(config["features"]["image_size"])
    percentile = float(config["features"]["curvature"]["edge_percentile"])
    records = []
    extractor.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            images = np.stack([load_gray(path, size) for path in rows.iloc[start:stop]["image_path"]])
            magnitude = extractor(torch.from_numpy(images[:, None]).to(device)).cpu().numpy()
            for local_index, gray in enumerate(images):
                winning_channel = np.argmax(magnitude[local_index], axis=2)
                curvature_map = bend_integer_labels[winning_channel].astype(np.float64)
                gradient = roberts(gray)
                threshold = np.percentile(gradient, percentile)
                edges = gradient > threshold
                edge_count = int(edges.sum())
                full_curvature = float(np.mean(curvature_map[edges])) if edge_count else np.nan
                weighted_edges = prf * edges
                edge_mass = float(weighted_edges.sum())
                prf_curvature = (
                    float(np.sum(weighted_edges * curvature_map) / edge_mass)
                    if edge_mass > 0 else np.nan
                )
                records.append({
                    "full__curvature": full_curvature,
                    "prf__curvature": prf_curvature,
                    "full__edge_fraction": float(edge_count / edges.size),
                    "prf__edge_mass": edge_mass,
                    "prf__effective_edge_pixels": float(
                        edge_mass**2 / np.sum(weighted_edges**2)
                    ) if np.sum(weighted_edges**2) > 0 else 0.0,
                })
    return pd.concat([identity_frame(rows), pd.DataFrame(records)], axis=1)
