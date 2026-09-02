from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from .utils import atomic_write_dataframe, output_root


def _subject_number(subject: str) -> int:
    return int(subject.removeprefix("S"))


def load_test_readout(config: dict, subject: str) -> dict:
    model_root = Path(config["paths"]["model_root"])
    model_dir = model_root / subject / config["test_model"]["model_dir"]
    with (model_dir / "best_weights.pkl").open("rb") as handle:
        weights = pickle.load(handle)
    with (model_root / subject / f"data_splits_{subject}.pkl").open("rb") as handle:
        splits = pickle.load(handle)
    set_id = int(config["test_model"]["set"])
    return {
        "model_dir": model_dir,
        "weights": weights,
        "mean": np.load(model_dir / "best_features_m.npy", mmap_mode="r"),
        "std": np.load(model_dir / "best_features_s.npy", mmap_mode="r"),
        "prf": np.load(model_dir / "best_prf_idx.npy", mmap_mode="r"),
        "val_ids": np.sort(np.asarray(splits[set_id]["val"], dtype=int)),
    }


def predict_validation(config: dict, subject: str, voxel_ids: list[int]) -> pd.DataFrame:
    readout = load_test_readout(config, subject)
    feature_root = Path(config["paths"]["feature_root"]) / subject / config["test_model"]["name"]
    beta_path = Path(config["paths"]["nsd_root"]) / "data" / f"{subject}_betas_avg_bigmask.hdf5"
    rows = []
    with h5py.File(beta_path, "r") as handle:
        for voxel_id in sorted(set(map(int, voxel_ids))):
            prf_id = int(readout["prf"][voxel_id])
            chunks = []
            for layer in config["test_model"]["layers"]:
                feature_path = feature_root / layer / f"features_prf_{prf_id}.npy"
                chunks.append(np.asarray(np.load(feature_path, mmap_mode="r")[readout["val_ids"]], dtype=np.float64))
            features = np.concatenate(chunks, axis=1)
            mean = np.asarray(readout["mean"][voxel_id], dtype=np.float64)
            std = np.asarray(readout["std"][voxel_id], dtype=np.float64)
            weights = np.asarray(readout["weights"][str(voxel_id)], dtype=np.float64).reshape(-1)
            if features.shape[1] != len(weights) - 1:
                raise ValueError(f"Feature/readout mismatch for voxel {voxel_id}")
            predictions = ((features - mean) / std) @ weights[:-1] + weights[-1]
            observed = np.asarray(handle["betas"][readout["val_ids"], voxel_id], dtype=np.float64)
            finite = np.isfinite(predictions) & np.isfinite(observed)
            residual = np.sum((observed[finite] - predictions[finite]) ** 2)
            total = np.sum((observed[finite] - np.mean(observed[finite])) ** 2)
            r2 = float(1.0 - residual / total) if total > 0 else np.nan
            for local_index, stimulus_id in enumerate(readout["val_ids"]):
                rows.append({
                    "subject": subject, "voxel_id": voxel_id,
                    "nsd_stimulus_id": int(stimulus_id), "validation_local_index": local_index,
                    "measured_response": float(observed[local_index]),
                    "predicted_response": float(predictions[local_index]),
                    "test_prf_id": prf_id, "test_validation_r2": r2,
                })
    return pd.DataFrame(rows)


def _top_finite(group: pd.DataFrame, column: str, count: int) -> pd.DataFrame:
    finite = group[np.isfinite(group[column])].sort_values(
        [column, "nsd_stimulus_id"], ascending=[False, True], kind="stable"
    ).head(count).copy()
    finite["image_rank"] = np.arange(1, len(finite) + 1)
    return finite


def export_nsd_images(config: dict, subject: str, stimulus_ids: list[int]) -> dict[int, str]:
    root = output_root(config) / "cache" / "nsd_images" / subject
    root.mkdir(parents=True, exist_ok=True)
    stimulus_path = Path(config["paths"]["nsd_root"]) / "stimuli" / f"{subject}_stimuli_224.h5py"
    paths: dict[int, str] = {}
    with h5py.File(stimulus_path, "r") as handle:
        for stimulus_id in sorted(set(map(int, stimulus_ids))):
            output = root / f"stimulus-{stimulus_id:06d}.png"
            if not output.is_file():
                pixels = np.asarray(handle["stimuli"][stimulus_id], dtype=np.uint8)
                if pixels.shape[0] == 3:
                    pixels = np.moveaxis(pixels, 0, -1)
                Image.fromarray(pixels, mode="RGB").save(output)
            paths[stimulus_id] = str(output)
    return paths


def select_nsd(config: dict, contexts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = output_root(config)
    all_predictions = []
    selections = []
    overlap_rows = []
    count = int(config["images_per_voxel"])
    for subject, subject_contexts in contexts.groupby("subject"):
        prediction = predict_validation(config, subject, subject_contexts["voxel_id"].tolist())
        all_predictions.append(prediction)
        selected_by_voxel: dict[int, dict[str, pd.DataFrame]] = {}
        for voxel_id, group in prediction.groupby("voxel_id"):
            measured = _top_finite(group, "measured_response", count)
            predicted = _top_finite(group, "predicted_response", count)
            selected_by_voxel[int(voxel_id)] = {"measured": measured, "predicted": predicted}
            measured_ids = set(measured["nsd_stimulus_id"])
            predicted_ids = set(predicted["nsd_stimulus_id"])
            overlap_rows.append({
                "subject": subject, "voxel_id": int(voxel_id),
                "intersection": len(measured_ids & predicted_ids),
                "union": len(measured_ids | predicted_ids),
                "jaccard": len(measured_ids & predicted_ids) / max(1, len(measured_ids | predicted_ids)),
            })
        selected_ids = [
            int(value) for groups in selected_by_voxel.values()
            for frame in groups.values() for value in frame["nsd_stimulus_id"]
        ]
        image_paths = export_nsd_images(config, subject, selected_ids)
        for context in subject_contexts.to_dict("records"):
            voxel_id = int(context["voxel_id"])
            for selection_name, condition_id in (
                ("measured", "nsd_measured_top"), ("predicted", "nsd_predicted_top")
            ):
                frame = selected_by_voxel[voxel_id][selection_name]
                for row in frame.to_dict("records"):
                    selections.append({
                        **context, "condition": condition_id, "condition_family": "nsd",
                        "image_rank": int(row["image_rank"]),
                        "image_path": image_paths[int(row["nsd_stimulus_id"])],
                        "seed": None, "generation_score": None, "ranking_score": None,
                        "nsd_stimulus_id": int(row["nsd_stimulus_id"]),
                        "measured_response": row["measured_response"],
                        "predicted_response": row["predicted_response"],
                        "test_prf_id": int(row["test_prf_id"]),
                        "test_validation_r2": row["test_validation_r2"],
                        "selection_prf_role": (
                            "no_model_prf_measured_response" if selection_name == "measured"
                            else "OPEN_CLIP_CONVNEXT_BASE_set2_own_prf"
                        ),
                    })
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    selections_frame = pd.DataFrame(selections)
    overlap_frame = pd.DataFrame(overlap_rows)
    atomic_write_dataframe(predictions_frame, root / "manifests" / "nsd_validation_predictions.csv")
    atomic_write_dataframe(selections_frame, root / "manifests" / "nsd_selected_images.csv")
    atomic_write_dataframe(overlap_frame, root / "summaries" / "nsd_selection_overlap.csv")
    return selections_frame, predictions_frame, overlap_frame
