"""Lightweight ranking definitions copied from the concatenated generation runner.

Kept separate so standalone fitting never imports diffusion or encoder packages.
Regression tests compare these definitions with the generation runner. The
generation runner itself is deliberately unchanged.
"""

import json
import os
import numpy as np

MODEL_PROVENANCE = {
    "CLIP_RN50": {
        "library": "clip",
        "architecture": "RN50",
        "pretrained": "OpenAI",
    },
    "OPEN_CLIP_RN50": {
        "library": "open_clip",
        "architecture": "RN50",
        "pretrained": "yfcc15m",
        "force_quick_gelu": True,
    },
    "DINO_RN50": {
        "library": "torch.hub",
        "architecture": "dino_resnet50",
        "repository": "facebookresearch/dino:main",
    },
    "SIMCLR_RN50": {
        "library": "huggingface_hub/torchvision",
        "architecture": "resnet50",
        "repository": "lightly-ai/simclrv1-imagenet1k-resnet50-1x",
        "checkpoint": "resnet50-1x.pth",
    },
    "ADV_RN50": {
        "library": "robustness/torchvision",
        "architecture": "resnet50",
        "checkpoint": "/ocean/projects/soc250009p/jzhao7/data/model/imagenet_l2_3_0.pt",
    },
    "OPEN_CLIP_CONVNEXT_BASE": {
        "library": "open_clip",
        "architecture": "convnext_base",
        "pretrained": "laion400m_s13b_b51k",
    },
}



class LayerSize:
    pass

setattr(LayerSize, "CLIP_RN50_layer_size", {
    'relu': [64, 112, 112],
    'relu1': [32, 112, 112],
    'relu2': [32, 112, 112],
    'relu3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_RN50_layer_size", {
    'relu': [64, 112, 112],
    'act1': [32, 112, 112],
    'act2': [32, 112, 112],
    'act3': [64, 112, 112],
    'avgpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "DINO_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "SIMCLR_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "ADV_RN50_layer_size", {
    'relu': [64, 112, 112],
    'maxpool': [64, 56, 56],
    'layer1': [256, 56, 56],
    'layer2': [512, 28, 28],
    'layer3': [1024, 14, 14],
    'layer4': [2048, 7, 7]
})

setattr(LayerSize, "OPEN_CLIP_CONVNEXT_BASE_layer_size", {
    'stem': [128, 56, 56],
    'stage1': [128, 56, 56],
    'stage2': [256, 28, 28],
    'stage3': [512, 14, 14],
    'stage4': [1024, 7, 7]
})

class Normalize:
    pass

setattr(Normalize, "CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_RN50_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_RN50_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])

setattr(Normalize, "DINO_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "DINO_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "SIMCLR_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "SIMCLR_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "ADV_RN50_MEAN", np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None])
setattr(Normalize, "ADV_RN50_STD", np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None])

setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_MEAN", np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None])
setattr(Normalize, "OPEN_CLIP_CONVNEXT_BASE_STD", np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None])


def load_nsd_rois(ss, args):
    roi_path = os.path.join(args.roi_path.format(ss))
    print(f"Loading rois from: {roi_path}")

    roi_info = np.load(roi_path, allow_pickle=True).item()
    noise_ceiling = roi_info['noise_ceiling_avgreps'] / 100.

    big_mask = roi_info['voxel_mask']
    roi_keys = ['roi_labels_kastner', 'roi_labels_retino', 'roi_labels_face', 'roi_labels_place', 'roi_labels_body']
    roi_names = ['kastner_atlas_roi_names', 'ret_prf_roi_names',  'floc_face_roi_names', 'floc_place_roi_names', 'floc_body_roi_names']
    roi_masks = dict()
    for key, name in zip(roi_keys, roi_names):
        roi_labels = roi_info[key][big_mask]
        roi_name = roi_info[name]
        for name in roi_name.keys():
            roi_masks[name] = roi_labels==roi_name[name]

    # combine ventral and dorsal retinotopic regions
    for name in ["V1", "V2", "V3"]:
        roi_masks[name] = roi_masks[name + "v"] | roi_masks[name + "d"]

    # combine FFA-1 and FFA-2
    roi_masks["FFA"] = roi_masks["FFA-1"] | roi_masks["FFA-2"]

    print(f"Loaded {len(roi_masks)} rois: {list(roi_masks.keys())}")

    return roi_masks, noise_ceiling


def load_concat_metadata(model_path, expected_model):
    """Load and validate the layer order used by a concatenated checkpoint."""
    metadata_path = os.path.join(model_path, "concat_metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    if metadata.get("model_name") != expected_model:
        raise ValueError(
            f"Checkpoint metadata model {metadata.get('model_name')!r} does "
            f"not match requested model {expected_model!r}: {metadata_path}"
        )
    if not metadata.get("layers"):
        raise ValueError(f"No concatenated layers recorded in {metadata_path}")
    return metadata


def validate_concat_layout(metadata, layer_sizes, description):
    """Ensure live encoder channels match the fitted concatenation layout."""
    expected_slices = {}
    start = 0
    for layer_name in metadata["layers"]:
        stop = start + int(layer_sizes[layer_name][0])
        expected_slices[layer_name] = [start, stop]
        start = stop
    if metadata.get("feature_slices") != expected_slices:
        raise ValueError(
            f"{description} feature layout mismatch: checkpoint has "
            f"{metadata.get('feature_slices')}, encoder requires "
            f"{expected_slices}"
        )
    if int(metadata.get("num_features", -1)) != start:
        raise ValueError(
            f"{description} metadata reports {metadata.get('num_features')} "
            f"features, but its layers contain {start}"
        )


def load_concatenated_features(features_model_folder, layer_names, prf_idx):
    """Load the same pRF ID from all layers and concatenate channel columns."""
    arrays = []
    num_images = None
    for layer_name in layer_names:
        feature_path = os.path.join(
            features_model_folder,
            layer_name,
            f"features_prf_{int(prf_idx)}.npy",
        )
        features = np.load(feature_path)
        if features.dtype.kind != 'f' or not np.all(np.isfinite(features)):
            raise ValueError(f"Expected finite floating-point features in {feature_path}")
        if features.ndim != 2:
            raise ValueError(
                f"Expected [images, channels] in {feature_path}, got "
                f"{features.shape}"
            )
        if num_images is None:
            num_images = features.shape[0]
        elif features.shape[0] != num_images:
            raise ValueError(f"Image count mismatch in {feature_path}")
        arrays.append(features)
    return np.concatenate(arrays, axis=1)


def save_online_ranking_model(
    output_folder,
    *,
    subject_id,
    region,
    voxel_id,
    voxel_index_in_region,
    voxel_rank,
    voxel_r2,
    generation_model_name,
    model_name,
    split_id,
    metadata,
    layer_sizes,
    prf_idx,
    prf_grid_name,
    prf_params,
    weights,
    intercept,
    feature_mean,
    feature_std,
    channel_indices,
    lambda_candidates,
    best_lambda_index,
    nested_sse,
    train_ids,
    val_ids,
    nest_ids,
    features_model_folder,
    ranking_model_path,
    data_splits_path,
):
    """Save a complete, per-voxel online ranking-model artifact."""
    os.makedirs(output_folder, exist_ok=True)

    parameters_filename = "parameters.npz"
    metadata_filename = "model.json"
    parameters_path = os.path.join(output_folder, parameters_filename)
    metadata_path = os.path.join(output_folder, metadata_filename)

    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    intercept_array = np.asarray(intercept, dtype=np.float64).reshape(())
    feature_mean_array = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
    feature_std_array = np.asarray(feature_std, dtype=np.float64).reshape(-1)
    channel_indices_array = np.asarray(channel_indices, dtype=np.int64).reshape(-1)
    lambda_candidates_array = np.asarray(
        lambda_candidates, dtype=np.float64
    ).reshape(-1)
    best_lambda_index = int(
        np.asarray(best_lambda_index).reshape(-1)[0]
    )
    nested_sse = float(np.asarray(nested_sse).reshape(-1)[0])

    num_features = int(metadata["num_features"])
    expected_shape = (num_features,)
    for array_name, array in (
        ("weights", weights_array),
        ("feature_mean", feature_mean_array),
        ("feature_std", feature_std_array),
        ("channel_indices", channel_indices_array),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"Online ranking {array_name} has shape {array.shape}; "
                f"expected {expected_shape}"
            )
    if not 0 <= best_lambda_index < len(lambda_candidates_array):
        raise IndexError(
            f"Best lambda index {best_lambda_index} is outside a grid of "
            f"length {len(lambda_candidates_array)}"
        )

    train_ids_array = np.asarray(train_ids, dtype=np.int64).reshape(-1)
    val_ids_array = np.asarray(val_ids, dtype=np.int64).reshape(-1)
    nest_ids_array = np.asarray(nest_ids, dtype=np.int64).reshape(-1)
    partial_parameters_path = parameters_path + ".partial.npz"
    np.savez_compressed(
        partial_parameters_path,
        weights=weights_array,
        intercept=intercept_array,
        feature_mean=feature_mean_array,
        feature_std=feature_std_array,
        channel_indices=channel_indices_array,
        lambda_candidates=lambda_candidates_array,
        best_lambda_index=np.asarray(best_lambda_index, dtype=np.int64),
        best_lambda=np.asarray(
            lambda_candidates_array[best_lambda_index], dtype=np.float64
        ),
        nested_sse=np.asarray(nested_sse, dtype=np.float64),
        train_ids=train_ids_array,
        validation_ids=val_ids_array,
        nested_ids=nest_ids_array,
    )

    layer_names = list(metadata["layers"])
    layer_layout = []
    for layer_name in layer_names:
        channels, height, width = (
            int(value) for value in layer_sizes[layer_name]
        )
        layer_layout.append({
            "name": layer_name,
            "channels": channels,
            "height": height,
            "width": width,
            "feature_slice": list(metadata["feature_slices"][layer_name]),
        })

    prf_x, prf_y, prf_sigma = (
        float(value) for value in np.asarray(prf_params).reshape(3)
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "braindive_online_voxel_ranking_model",
        "subject": int(subject_id),
        "region": str(region),
        "voxel": {
            "id": int(voxel_id),
            "index_in_region": int(voxel_index_in_region),
            "generation_model_r2_rank": int(voxel_rank),
            "generation_model_r2": float(voxel_r2),
        },
        "ranking_encoder": {
            "name": model_name,
            "online_voxel_readout_fit": True,
            "pretrained_backbone_parameters_included": False,
            "provenance": MODEL_PROVENANCE.get(model_name, {}),
            "checkpoint_folder": os.path.abspath(ranking_model_path),
            "checkpoint_concat_metadata": metadata,
            "input": {
                "color_space": "RGB",
                "range": [0.0, 1.0],
                "resize": [224, 224],
                "resize_mode": "bilinear",
                "antialias": True,
                "mean": getattr(
                    Normalize, f"{model_name}_MEAN"
                ).reshape(-1).astype(float).tolist(),
                "std": getattr(
                    Normalize, f"{model_name}_STD"
                ).reshape(-1).astype(float).tolist(),
            },
            "concatenation_order": layer_names,
            "layers": layer_layout,
            "num_features": num_features,
        },
        "prf": {
            "source": "generation_model_best_prf_idx",
            "generation_model_name": generation_model_name,
            "shared_with_generation_model": True,
            "grid_name": prf_grid_name,
            "id": int(prf_idx),
            "x_aperture_units": prf_x,
            "y_aperture_units": prf_y,
            "sigma_aperture_units": prf_sigma,
            "nsd_aperture_degrees": 8.4,
            "x_degrees": prf_x * 8.4,
            "y_degrees": prf_y * 8.4,
            "sigma_degrees": prf_sigma * 8.4,
            "pooling": "normalized_isotropic_gaussian_weighted_mean",
            "rasterized_at_each_layer_native_resolution": True,
        },
        "fit": {
            "model_type": "ridge_regression_with_intercept",
            "target": "NSD averaged beta response for this voxel",
            "split_id": int(split_id),
            "feature_normalization": "per-channel z-score",
            "normalization_statistics_partition": "train_plus_nested",
            "weights_fit_partition": "train",
            "lambda_selection_partition": "nested",
            "validation_partition_used_during_fit": False,
            "intercept_in_ridge_penalty": True,
            "ridge_solver_epsilon": 1e-4,
            "best_lambda_index": best_lambda_index,
            "best_lambda": float(
                lambda_candidates_array[best_lambda_index]
            ),
            "best_nested_sse": nested_sse,
            "partition_counts": {
                "train": int(len(train_ids_array)),
                "validation": int(len(val_ids_array)),
                "nested": int(len(nest_ids_array)),
            },
            "prediction_equation": (
                "score = dot((features - feature_mean) / feature_std, "
                "weights) + intercept"
            ),
        },
        "source_data": {
            "precomputed_feature_folder": os.path.abspath(
                features_model_folder
            ),
            "data_splits_file": os.path.abspath(data_splits_path),
        },
        "artifacts": {
            "parameters_file": parameters_filename,
            "parameter_arrays": {
                "weights": "float64[num_features]",
                "intercept": "float64 scalar",
                "feature_mean": "float64[num_features]",
                "feature_std": "float64[num_features]",
                "channel_indices": "int64[num_features]",
                "lambda_candidates": "float64[num_lambdas]",
                "best_lambda_index": "int64 scalar",
                "best_lambda": "float64 scalar",
                "nested_sse": "float64 scalar",
                "train_ids": "int64[num_train_images]",
                "validation_ids": "int64[num_validation_images]",
                "nested_ids": "int64[num_nested_images]",
            },
        },
    }
    partial_metadata_path = metadata_path + ".partial"
    with open(partial_metadata_path, "w") as metadata_handle:
        json.dump(payload, metadata_handle, indent=2, allow_nan=False)
    os.replace(partial_parameters_path, parameters_path)
    os.replace(partial_metadata_path, metadata_path)

    return metadata_path, parameters_path


def rank_model_fitting(voxel_id, best_prf_idx, voxel_data, train_ids, val_ids, nest_ids, features_model_folder, layer_names, device):
    """Fit one concatenated-feature ridge readout with the original solver."""
    import torch
    import model_fitting_utils

    train_voxel = voxel_data[train_ids, voxel_id]
    nest_voxel = voxel_data[nest_ids, voxel_id]

    train_voxel = torch.from_numpy(train_voxel).float().unsqueeze(1).to(device)
    nest_voxel = torch.from_numpy(nest_voxel).float().unsqueeze(1).to(device)

    features = load_concatenated_features(
        features_model_folder, layer_names, best_prf_idx
    )

    # split train and nest features, then z-score the features across images axis 0 using model_fitting_utils.split_normalize_feats
    # train_features, val_features, nest_features, features_s, features_m = model_fitting_utils.split_normalize_feats(features, train_ids, val_ids, nest_ids)
    train_features, val_features, nest_features, features_s, features_m = (
        model_fitting_utils.split_normalize_feats(
            features, train_ids, val_ids, nest_ids
        )
    )
    channel_kept = np.arange(features.shape[1], dtype=int)
    # train_features, val_features, nest_features, keep_idx = model_fitting_utils.split_feats(features, train_ids, val_ids, nest_ids)

    # add the intercept: a column of ones
    train_features = np.concatenate([train_features, np.ones(shape=(len(train_features), 1), dtype=train_features.dtype)], axis=1)
    nest_features = np.concatenate([nest_features, np.ones(shape=(len(nest_features), 1), dtype=nest_features.dtype)], axis=1)

    train_features = torch.from_numpy(train_features).float().to(device)
    nest_features = torch.from_numpy(nest_features).float().to(device)
    n_lambdas = 20
    small_value = 0.0001
    lambdas = np.logspace(np.log(small_value),np.log(10**10+small_value),n_lambdas, \
                        dtype=np.float32, base=np.e) - small_value

    best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)
    # best_weights, best_lambda_idx, best_nest_loss = model_fitting_utils.solve_ridge_svd(train_features, train_voxel, nest_features, nest_voxel, lambdas, eps=1e-4, return_loss=True)


    return (
        best_weights,
        channel_kept,
        best_lambda_idx,
        best_nest_loss,
        lambdas,
        features_s,
        features_m,
    )
