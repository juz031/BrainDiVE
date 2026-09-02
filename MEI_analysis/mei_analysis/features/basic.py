from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.color import rgb2hsv, rgb2lab

from ..utils import load_rgb
from .common import identity_frame


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.sum(weights * values))
    std = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
    return mean, std


def extract_basic(rows: pd.DataFrame, prf: np.ndarray, image_size: int) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        rgb = load_rgb(row.image_path, image_size)
        lab = rgb2lab(rgb.astype(np.float32) / 255.0)
        saturation = rgb2hsv(rgb.astype(np.float32) / 255.0)[..., 1]
        record = {}
        for label, values in (
            ("L", lab[..., 0]), ("a", lab[..., 1]), ("b", lab[..., 2]),
            ("saturation", saturation),
        ):
            record[f"full__mean_{label}"] = float(np.mean(values))
            record[f"full__std_{label}"] = float(np.std(values))
            mean, std = weighted_mean_std(values, prf)
            record[f"prf__mean_{label}"] = mean
            record[f"prf__std_{label}"] = std
        records.append(record)
    return pd.concat([identity_frame(rows), pd.DataFrame(records)], axis=1)
