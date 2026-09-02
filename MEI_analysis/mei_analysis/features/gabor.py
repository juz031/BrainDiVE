from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils import load_gray
from .common import identity_frame


def _legacy_gabor(config: dict):
    legacy = str(Path(config["paths"]["legacy_image_stats"]))
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    import gabor_model
    return gabor_model


def build_extractor(config: dict, device: torch.device):
    settings = config["features"]["gabor"]
    size = int(config["features"]["image_size"])
    frequencies = np.asarray(settings["spatial_frequencies_cycles_per_image"], dtype=float)
    orientations = np.asarray(settings["orientations_degrees"], dtype=float)
    model = _legacy_gabor(config)
    bank = model.filter_bank(
        orients_deg=orientations, freqs_cpp=frequencies / size,
        spat_freq_bw=float(settings["spatial_frequency_bandwidth"]),
        n_sd_out=int(settings["n_sd_out"]),
        spat_aspect_ratio=float(settings["spatial_aspect_ratio"]),
        image_size=size, padding_mode=settings["padding_mode"],
    )
    return model.gabor_feature_extractor_spat(bank, device=device), frequencies, orientations


def pool_response_maps(magnitude: torch.Tensor, prf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool already-filtered response maps over the full image and the pRF."""
    if magnitude.ndim != 4 or prf.ndim != 2:
        raise ValueError("Expected magnitude [batch,height,width,features] and pRF [height,width]")
    if tuple(magnitude.shape[1:3]) != tuple(prf.shape):
        raise ValueError(f"Response/pRF spatial mismatch: {tuple(magnitude.shape)} vs {tuple(prf.shape)}")
    return magnitude.mean(dim=(1, 2)), torch.einsum("bhwf,hw->bf", magnitude, prf)


def extract_gabor(
    config: dict, rows: pd.DataFrame, prf: np.ndarray,
    device: torch.device, batch_size: int = 4,
) -> pd.DataFrame:
    extractor, frequencies, orientations = build_extractor(config, device)
    size = int(config["features"]["image_size"])
    feature_names = [
        f"gabor_sf{sf_index:02d}_ori{ori_index:02d}"
        for sf_index in range(len(frequencies)) for ori_index in range(len(orientations))
    ]
    full_values = np.empty((len(rows), len(feature_names)), dtype=np.float32)
    prf_values = np.empty_like(full_values)
    kernel = torch.as_tensor(prf, dtype=torch.float32, device=device)
    extractor.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            images = np.stack([load_gray(path, size) for path in rows.iloc[start:stop]["image_path"]])
            tensor = torch.from_numpy(images[:, None]).to(device)
            magnitude, _ = extractor(tensor)
            full, weighted = pool_response_maps(magnitude, kernel)
            full_values[start:stop] = full.cpu().numpy()
            prf_values[start:stop] = weighted.cpu().numpy()
    features = {}
    for index, name in enumerate(feature_names):
        features[f"full__{name}"] = full_values[:, index]
        features[f"prf__{name}"] = prf_values[:, index]
    return pd.concat([identity_frame(rows), pd.DataFrame(features)], axis=1)


def feature_info(config: dict) -> pd.DataFrame:
    settings = config["features"]["gabor"]
    rows = []
    for sf_index, frequency in enumerate(settings["spatial_frequencies_cycles_per_image"]):
        for ori_index, orientation in enumerate(settings["orientations_degrees"]):
            rows.append({
                "feature": f"gabor_sf{sf_index:02d}_ori{ori_index:02d}",
                "spatial_frequency_index": sf_index,
                "spatial_frequency_cycles_per_image": float(frequency),
                "orientation_index": ori_index, "orientation_degrees": float(orientation),
            })
    return pd.DataFrame(rows)
