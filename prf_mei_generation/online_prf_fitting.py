"""Fit cross-backbone voxel readouts while keeping the generator's pRF fixed.

The precomputed NSD feature files contain one pooled feature matrix per pRF.
For cross-backbone ranking we deliberately load the pRF selected by the
generator checkpoint, then fit only the linear ridge readout of the ranking
backbone.  This prevents a backbone change from also moving the receptive
field to a different image location.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import tables
import torch

from max_activation_utils import VoxelTarget


DEFAULT_FEATURE_ROOT = Path("/user_data/junruz/prf_features/S1")
DEFAULT_SPLIT_PATH = Path(
    "/user_data/junruz/prf_models/fixed/split_1_zscore/S1/"
    "data_splits_S1.pkl"
)
DEFAULT_BETA_PATH = Path(
    "/lab_data/hendersonlab/datasets/nsd_preproc/data/"
    "S1_betas_avg_bigmask.hdf5"
)


@dataclass
class FixedPrfReadout:
    """A fitted target plus provenance and held-out fit diagnostics."""

    target: VoxelTarget
    metadata: dict[str, Any]
    cache_path: Path


def ridge_lambda_grid() -> np.ndarray:
    """Return the exact lambda grid used by the fixed split/z-score training."""
    small_value = 0.0001
    return (
        np.logspace(
            np.log(small_value),
            np.log(10**10 + small_value),
            20,
            dtype=np.float64,
            base=np.e,
        )
        - small_value
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - residual / total) if total > 0.0 else float("nan")


class FixedPrfReadoutFitter:
    """Fit and cache voxel readouts for one backbone/layer/data split."""

    CACHE_VERSION = 1

    def __init__(
        self,
        model_name: str,
        layer_name: str,
        split_id: int,
        device: torch.device,
        cache_dir: Path,
        *,
        feature_root: Path = DEFAULT_FEATURE_ROOT,
        split_path: Path = DEFAULT_SPLIT_PATH,
        beta_path: Path = DEFAULT_BETA_PATH,
    ) -> None:
        if split_id not in {1, 2}:
            raise ValueError("split_id must be 1 or 2.")
        self.model_name = model_name
        self.layer_name = layer_name
        self.split_id = split_id
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.feature_dir = Path(feature_root) / model_name / layer_name
        self.split_path = Path(split_path)
        self.beta_path = Path(beta_path)
        self.splits = self._load_splits()

    def _load_splits(self) -> dict[str, np.ndarray]:
        with open(self.split_path, "rb") as handle:
            saved_splits = pickle.load(handle)
        split = saved_splits[self.split_id]
        required = {"train", "val", "nest"}
        if not required.issubset(split):
            raise ValueError(
                f"{self.split_path} split {self.split_id} lacks {required}."
            )
        result = {
            name: np.asarray(split[name], dtype=np.int64)
            for name in required
        }
        all_ids = np.concatenate(list(result.values()))
        if len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("train/val/nest image IDs must be disjoint.")
        return result

    def cache_path(self, voxel_idx: int, prf_idx: int) -> Path:
        return (
            self.cache_dir
            / f"{self.model_name}_set{self.split_id}"
            / self.layer_name
            / f"fixed_prf_{prf_idx:04d}"
            / f"voxel_{voxel_idx:06d}.pt"
        )

    def feature_path(self, prf_idx: int) -> Path:
        return self.feature_dir / f"features_prf_{prf_idx}.npy"

    def fit_targets(
        self,
        voxel_to_prf: Mapping[int, int],
        *,
        force: bool = False,
    ) -> dict[int, FixedPrfReadout]:
        """Fit requested voxels, grouping shared pRFs to avoid repeated work."""
        requested = {
            int(voxel_idx): int(prf_idx)
            for voxel_idx, prf_idx in voxel_to_prf.items()
        }
        if not requested:
            return {}

        fitted: dict[int, FixedPrfReadout] = {}
        missing: dict[int, int] = {}
        for voxel_idx, prf_idx in requested.items():
            path = self.cache_path(voxel_idx, prf_idx)
            if path.is_file() and not force:
                fitted[voxel_idx] = self._load_cached(path)
            else:
                missing[voxel_idx] = prf_idx

        if not missing:
            return fitted

        missing_order = list(missing)
        responses = self.load_responses(missing_order)
        response_column = {
            voxel_idx: column
            for column, voxel_idx in enumerate(missing_order)
        }
        prf_groups: dict[int, list[int]] = {}
        for voxel_idx, prf_idx in missing.items():
            prf_groups.setdefault(prf_idx, []).append(voxel_idx)

        for prf_idx, voxel_indices in sorted(prf_groups.items()):
            group_responses = responses[
                :,
                [response_column[voxel_idx] for voxel_idx in voxel_indices],
            ]
            group_fits = self._fit_prf_group(
                prf_idx,
                voxel_indices,
                group_responses,
            )
            fitted.update(group_fits)

        return {voxel_idx: fitted[voxel_idx] for voxel_idx in requested}

    def load_responses(
        self,
        voxel_indices: Sequence[int],
        image_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Read only requested beta columns, preserving the caller's order."""
        if not voxel_indices:
            return np.empty((0, 0), dtype=np.float64)
        sorted_indices = sorted(set(int(index) for index in voxel_indices))
        with tables.open_file(self.beta_path, mode="r") as handle:
            dataset = handle.root.betas
            if sorted_indices[-1] >= dataset.shape[1]:
                raise IndexError(
                    f"voxel {sorted_indices[-1]} exceeds beta width "
                    f"{dataset.shape[1]}."
                )
            sorted_values = np.asarray(
                dataset[:, sorted_indices],
                dtype=np.float64,
            )
        position = {voxel_idx: i for i, voxel_idx in enumerate(sorted_indices)}
        values = sorted_values[
            :,
            [position[int(voxel_idx)] for voxel_idx in voxel_indices],
        ]
        if image_ids is not None:
            values = values[np.asarray(image_ids, dtype=np.int64)]
        return values

    def _fit_prf_group(
        self,
        prf_idx: int,
        voxel_indices: Sequence[int],
        responses: np.ndarray,
    ) -> dict[int, FixedPrfReadout]:
        feature_path = self.feature_path(prf_idx)
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got {features.shape}."
            )
        if features.shape[0] != responses.shape[0]:
            raise ValueError(
                f"Feature/response image mismatch: {features.shape[0]} vs "
                f"{responses.shape[0]}."
            )

        train_ids = self.splits["train"]
        nest_ids = self.splits["nest"]
        val_ids = self.splits["val"]

        # Match the original fixed split/z-score pipeline: feature statistics
        # use train+nest images, while ridge weights use train responses only.
        normalization_ids = np.concatenate([train_ids, nest_ids])
        normalization_features = np.asarray(
            features[normalization_ids],
            dtype=np.float64,
        )
        features_m = normalization_features.mean(axis=0)
        features_s = normalization_features.std(axis=0) + 1e-12
        del normalization_features

        def normalized_rows(image_ids: np.ndarray) -> np.ndarray:
            rows = np.asarray(features[image_ids], dtype=np.float64)
            rows -= features_m
            rows /= features_s
            return np.concatenate(
                [rows, np.ones((len(rows), 1), dtype=np.float64)],
                axis=1,
            )

        x_train = normalized_rows(train_ids)
        x_nest = normalized_rows(nest_ids)
        x_val = normalized_rows(val_ids)
        y_train = responses[train_ids]
        y_nest = responses[nest_ids]
        y_val = responses[val_ids]

        weights, lambda_indices, nest_sse = self._solve_ridge_grid(
            x_train,
            y_train,
            x_nest,
            y_nest,
        )
        val_predictions = x_val @ weights
        lambdas = ridge_lambda_grid()

        result: dict[int, FixedPrfReadout] = {}
        for column, voxel_idx in enumerate(voxel_indices):
            weights_with_intercept = weights[:, column].astype(
                np.float32,
                copy=False,
            )
            prediction = val_predictions[:, column]
            metadata = {
                "cache_version": self.CACHE_VERSION,
                "fit_policy": "generator_fixed_prf_online_ridge",
                "model_name": self.model_name,
                "layer_name": self.layer_name,
                "split_id": self.split_id,
                "voxel_idx": int(voxel_idx),
                "prf_idx": int(prf_idx),
                "selected_lambda": float(lambdas[lambda_indices[column]]),
                "selected_lambda_index": int(lambda_indices[column]),
                "nest_sse": float(nest_sse[column]),
                "val_r2": _r2(y_val[:, column], prediction),
                "val_correlation": _correlation(
                    y_val[:, column],
                    prediction,
                ),
                "n_train": int(len(train_ids)),
                "n_nest": int(len(nest_ids)),
                "n_val": int(len(val_ids)),
                "feature_path": str(feature_path),
                "beta_path": str(self.beta_path),
                "split_path": str(self.split_path),
            }
            payload = {
                "metadata": metadata,
                "weights_with_intercept": weights_with_intercept,
                "features_m": features_m.astype(np.float32, copy=False),
                "features_s": features_s.astype(np.float32, copy=False),
            }
            path = self.cache_path(voxel_idx, prf_idx)
            _atomic_torch_save(payload, path)
            result[int(voxel_idx)] = self._payload_to_readout(
                payload,
                path,
            )
        return result

    def _solve_ridge_grid(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_nest: np.ndarray,
        y_nest: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select ridge lambda on nest data using one eigendecomposition."""
        x_train_t = torch.as_tensor(
            x_train,
            dtype=torch.float64,
            device=self.device,
        )
        y_train_t = torch.as_tensor(
            y_train,
            dtype=torch.float64,
            device=self.device,
        )
        x_nest_t = torch.as_tensor(
            x_nest,
            dtype=torch.float64,
            device=self.device,
        )
        y_nest_t = torch.as_tensor(
            y_nest,
            dtype=torch.float64,
            device=self.device,
        )

        xtx = x_train_t.T @ x_train_t
        xty = x_train_t.T @ y_train_t
        eigenvalues, eigenvectors = torch.linalg.eigh(xtx)
        eigenvalues = eigenvalues.clamp_min(0.0)
        projected_targets = eigenvectors.T @ xty
        lambdas = torch.as_tensor(
            ridge_lambda_grid(),
            dtype=torch.float64,
            device=self.device,
        )
        denominator = (
            eigenvalues[None, :, None] + lambdas[:, None, None] + 1e-4
        )
        all_weights = torch.einsum(
            "ij,ljv->liv",
            eigenvectors,
            projected_targets[None, :, :] / denominator,
        )
        nest_predictions = torch.einsum(
            "nd,ldv->nlv",
            x_nest_t,
            all_weights,
        )
        nest_sse = ((nest_predictions - y_nest_t[:, None, :]) ** 2).sum(
            dim=0
        )
        best_indices = nest_sse.argmin(dim=0)
        selected_weights = torch.stack(
            [
                all_weights[best_indices[column], :, column]
                for column in range(y_train.shape[1])
            ],
            dim=1,
        )
        selected_sse = nest_sse[
            best_indices,
            torch.arange(y_train.shape[1], device=self.device),
        ]
        return (
            selected_weights.cpu().numpy(),
            best_indices.cpu().numpy(),
            selected_sse.cpu().numpy(),
        )

    def _load_cached(self, path: Path) -> FixedPrfReadout:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        expected = {
            "cache_version": self.CACHE_VERSION,
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "split_id": self.split_id,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"Cached readout mismatch at {path}: "
                    f"{key}={metadata.get(key)!r}, expected {value!r}."
                )
        return self._payload_to_readout(payload, path)

    def _payload_to_readout(
        self,
        payload: dict[str, Any],
        path: Path,
    ) -> FixedPrfReadout:
        metadata = payload["metadata"]
        weights = np.asarray(
            payload["weights_with_intercept"],
            dtype=np.float32,
        )
        target = VoxelTarget(
            voxel_idx=int(metadata["voxel_idx"]),
            prf_idx=int(metadata["prf_idx"]),
            weights=torch.as_tensor(weights[:-1], device=self.device),
            intercept=torch.as_tensor(weights[-1], device=self.device),
            features_m=torch.as_tensor(
                np.asarray(payload["features_m"], dtype=np.float32),
                device=self.device,
            ),
            features_s=torch.as_tensor(
                np.asarray(payload["features_s"], dtype=np.float32),
                device=self.device,
            ),
            channel_kept=None,
        )
        return FixedPrfReadout(
            target=target,
            metadata=dict(metadata),
            cache_path=path,
        )

    def predict_natural_images(
        self,
        readout: FixedPrfReadout,
        image_ids: Sequence[int],
    ) -> np.ndarray:
        """Score natural images from their already-pooled pRF features."""
        ids = np.asarray(image_ids, dtype=np.int64)
        features = np.load(
            self.feature_path(readout.target.prf_idx),
            mmap_mode="r",
        )
        rows = np.asarray(features[ids], dtype=np.float32)
        mean = readout.target.features_m.detach().cpu().numpy()
        std = readout.target.features_s.detach().cpu().numpy()
        weights = readout.target.weights.detach().cpu().numpy()
        intercept = float(readout.target.intercept.detach().cpu())
        return ((rows - mean) / std) @ weights + intercept
