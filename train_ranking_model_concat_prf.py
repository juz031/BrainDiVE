"""Train per-voxel ranking readouts at generation-model-selected pRFs.

No diffusion model, live encoder, feature extraction, or pRF search is used.
See script/b2/README_ranking_training.md for CLI, output and resume semantics.
Only load trusted local NSD/checkpoint files: their existing formats use pickle.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
import time
from types import SimpleNamespace

import h5py
import numpy as np

from ranking_model_training_utils import (
    LayerSize, load_concat_metadata, load_nsd_rois, rank_model_fitting,
    save_online_ranking_model, validate_concat_layout,
)
from voxel_selection import select_voxels_by_rank


PROJECT_ROOT = Path('/ocean/projects/soc250009p/jzhao7')
MODELS = ('DINO_RN50', 'OPEN_CLIP_RN50', 'ADV_RN50',
          'OPEN_CLIP_CONVNEXT_BASE', 'CLIP_RN50', 'SIMCLR_RN50')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject_id', type=int, required=True, choices=range(1, 9))
    parser.add_argument('--gen_model', choices=MODELS, default='DINO_RN50')
    parser.add_argument('--rank_model', choices=MODELS, default='OPEN_CLIP_RN50')
    parser.add_argument('--split_id', type=int, choices=(1, 2), default=1)
    parser.add_argument('--roi', nargs='+', default=['hV4'])
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--voxel_rank_range', nargs=2, type=int, metavar=('START', 'END'))
    parser.add_argument('--r2_noise_ceiling_threshold', type=float, default=0.)
    parser.add_argument('--nsd_root', type=Path,
                        default=Path('/ocean/projects/soc250009p/shared/datasets/nsd_preproc'))
    parser.add_argument('--roi_path', type=Path,
                        help='Defaults to NSD_ROOT/rois/S<ID>_voxel_roi_info.npy')
    parser.add_argument('--model_root', type=Path,
                        default=PROJECT_ROOT / 'data/model/concat/split_1_zscore')
    parser.add_argument('--feature_root', type=Path,
                        default=PROJECT_ROOT / 'data/prf_features')
    parser.add_argument('--save_root', type=Path,
                        default=PROJECT_ROOT / 'data/model/ranking')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument('--skip_existing', action='store_true',
                          help='Skip only complete artifacts with matching inputs/configuration')
    existing.add_argument('--overwrite', action='store_true',
                          help='Explicitly replace the selected existing artifacts')
    parser.add_argument('--dry_run', action='store_true',
                        help='Check selection, splits and feature headers; no training or writes')
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error('--k must be positive')
    if args.voxel_rank_range:
        start, end = args.voxel_rank_range
        if not 1 <= start <= end:
            parser.error('--voxel_rank_range requires 1 <= START <= END')
    if not np.isfinite(args.r2_noise_ceiling_threshold):
        parser.error('--r2_noise_ceiling_threshold must be finite')
    if len(set(args.roi)) != len(args.roi):
        parser.error('--roi must not contain duplicates')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', roi) for roi in args.roi):
        parser.error('ROI names must be safe single path components')
    if args.roi_path is None:
        args.roi_path = args.nsd_root / 'rois' / f'S{args.subject_id}_voxel_roi_info.npy'
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path, *, content_hash=False):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f'Missing or empty input file: {path}')
    result = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if content_hash:
        result['sha256'] = sha256_file(path)
    return result


def validate_splits(ids, image_count):
    result = []
    for name in ('train', 'val', 'nest'):
        values = np.asarray(ids[name])
        if values.ndim != 1 or values.size == 0 or values.dtype.kind not in 'iu':
            raise ValueError(f'Split {name} must be a nonempty 1-D integer array')
        if np.any(values < 0) or np.any(values >= image_count):
            raise ValueError(f'Split {name} contains out-of-range image indices')
        result.append(values.astype(np.int64))
    combined = np.concatenate(result)
    if len(np.unique(combined)) != len(combined):
        raise ValueError('Train/validation/nested splits overlap or contain duplicate indices')
    return tuple(result)


def select_records(args, roi_masks, noise_ceiling, r2, best_prf_idx):
    """Reconstruct evaluation order, then translate ranks to big-mask indices."""
    nc = np.asarray(noise_ceiling)
    prfs = np.asarray(best_prf_idx).reshape(-1)
    if nc.ndim != 1 or len(prfs) != len(nc):
        raise ValueError('Noise ceiling and generation pRF arrays must share big-mask ordering')
    records = []
    for region in args.roi:
        if region not in roi_masks or region not in r2:
            raise ValueError(f'ROI {region!r} is missing from the atlas or generation R² file')
        mask = np.asarray(roi_masks[region])
        if mask.shape != nc.shape or mask.dtype.kind != 'b':
            raise ValueError(f'Invalid big-mask ROI shape/type for {region}')
        selected = mask & (nc > args.r2_noise_ceiling_threshold)
        voxel_ids = np.flatnonzero(selected)
        scores = np.asarray(r2[region]['r2_voxels']).reshape(-1)
        if scores.size != voxel_ids.size:
            raise ValueError(f'R²/voxel index mismatch for {region}: {scores.size} scores '
                             f'vs {voxel_ids.size} voxels; check atlas and noise-ceiling threshold')
        if not np.all(np.isfinite(scores)):
            raise ValueError(f'Nonfinite R² values in {region}; ranking would be ambiguous')
        if 'noise_ceiling' not in r2[region]:
            raise ValueError(f'No saved noise_ceiling for {region}; cannot verify evaluation ordering')
        saved_nc = np.asarray(r2[region]['noise_ceiling']).reshape(-1)
        if saved_nc.shape != nc[selected].shape or not np.allclose(
                saved_nc, nc[selected], equal_nan=True):
            raise ValueError(f'Noise-ceiling ordering mismatch for {region}')
        indices, values, ranks = select_voxels_by_rank(scores, args.k, args.voxel_rank_range)
        for idx, score, rank in zip(indices, values, ranks):
            voxel_id = int(voxel_ids[idx])
            prf = prfs[voxel_id]
            if not np.isfinite(prf) or prf != int(prf):
                raise ValueError(f'Invalid generation pRF ID for voxel {voxel_id}: {prf}')
            records.append(dict(region=region, voxel_id=voxel_id, index_in_region=int(idx),
                                voxel_rank=int(rank), voxel_r2=float(score), prf_id=int(prf)))
    return records


def output_folder(args, record):
    return (args.save_root / f'S{args.subject_id}' /
            f'gen-{args.gen_model}__rank-{args.rank_model}' / f'split-{args.split_id:02d}' /
            f'roi-{record["region"]}' /
            f'voxel-rank-{record["voxel_rank"]:02d}__id-{record["voxel_id"]:06d}')


def prepare_inputs(args):
    """Read small metadata and array headers, not full features or all-voxel betas."""
    from prf_utils import get_prf_grid

    subject_folder = args.model_root / f'S{args.subject_id}'
    gen_folder = subject_folder / f'{args.gen_model}_set{args.split_id}'
    rank_folder = subject_folder / f'{args.rank_model}_set{args.split_id}'
    feature_folder = args.feature_root / f'S{args.subject_id}' / args.rank_model
    gen_meta = load_concat_metadata(gen_folder, args.gen_model)
    rank_meta = load_concat_metadata(rank_folder, args.rank_model)
    for name, metadata in ((args.gen_model, gen_meta), (args.rank_model, rank_meta)):
        sizes = getattr(LayerSize, f'{name}_layer_size')
        if len(set(metadata['layers'])) != len(metadata['layers']):
            raise ValueError(f'Duplicate concatenated layers for {name}')
        if set(metadata['layers']) - set(sizes):
            raise ValueError(f'Unknown concatenated layers for {name}')
        validate_concat_layout(metadata, sizes, name)
    paths = dict(
        roi=args.roi_path,
        generation_metadata=gen_folder / 'concat_metadata.json',
        ranking_metadata=rank_folder / 'concat_metadata.json',
        generation_r2=gen_folder / 'r2.pkl',
        generation_prf=gen_folder / 'best_prf_idx.npy',
        splits=subject_folder / f'data_splits_S{args.subject_id}.pkl',
        betas=args.nsd_root / 'data' / f'S{args.subject_id}_betas_avg_bigmask.hdf5',
        labels=args.nsd_root / 'labels' / f'S{args.subject_id}_image_info.csv',
    )
    identities = {name: file_identity(path, content_hash=(name != 'betas'))
                  for name, path in paths.items()}
    roi_masks, noise_ceiling = load_nsd_rois(
        args.subject_id, SimpleNamespace(roi_path=str(args.roi_path)))
    with open(paths['generation_r2'], 'rb') as handle:
        r2 = pickle.load(handle)
    best_prf_idx = np.load(paths['generation_prf'], allow_pickle=False)
    with open(paths['labels'], newline='') as handle:
        rows = list(csv.DictReader(handle))
    repetitions = np.array([float(row['n_reps']) for row in rows])
    with h5py.File(paths['betas'], 'r') as handle:
        betas = handle['betas']
        if betas.ndim != 2 or betas.shape[1] != len(noise_ceiling):
            raise ValueError('Beta columns do not match the ROI big-mask voxel count')
        image_count = betas.shape[0]
        if len(rows) != image_count:
            raise ValueError('Image-label rows do not match beta rows')
        good = ~np.isnan(betas[:, 0])
    # Never drop response rows without identically remapping features and splits.
    if not np.all(good) or not np.all(np.isfinite(repetitions) & (repetitions > 0)):
        raise ValueError('Missing image responses: refusing to silently misalign feature/split rows')
    with open(paths['splits'], 'rb') as handle:
        splits = validate_splits(pickle.load(handle)[args.split_id], image_count)
    records = select_records(args, roi_masks, noise_ceiling, r2, best_prf_idx)
    grid, grid_name = get_prf_grid('default-log-polar')
    gen_prfs, rank_prfs = set(gen_meta['prf_ids']), set(rank_meta['prf_ids'])
    feature_inputs = {}
    for prf in sorted({record['prf_id'] for record in records}):
        if prf not in gen_prfs or prf not in rank_prfs or not 0 <= prf < len(grid):
            raise ValueError(f'Generation pRF {prf} is not in both metadata grids')
        feature_inputs[prf] = []
        for layer in rank_meta['layers']:
            path = feature_folder / layer / f'features_prf_{prf}.npy'
            identity = file_identity(path)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            start, stop = rank_meta['feature_slices'][layer]
            if array.shape != (image_count, stop - start) or array.dtype.kind != 'f':
                raise ValueError(f'Feature shape/dtype mismatch: {path}: {array.shape}, {array.dtype}')
            identity.update(shape=list(array.shape), dtype=str(array.dtype))
            feature_inputs[prf].append(identity)
            del array
    source_dir = Path(__file__).resolve().parent
    implementation = {name: sha256_file(source_dir / name) for name in (
        'train_ranking_model_concat_prf.py', 'ranking_model_training_utils.py',
        'model_fitting_utils.py', 'prf_utils.py', 'voxel_selection.py')}
    common = dict(schema_version=1, subject=args.subject_id, gen_model=args.gen_model,
                  rank_model=args.rank_model, split_id=args.split_id,
                  r2_noise_ceiling_threshold=args.r2_noise_ceiling_threshold,
                  roi_policy='retino_over_kastner; V1-3=ventral+dorsal; FFA=FFA-1+FFA-2',
                  input_files=identities, implementation_sha256=implementation)
    return dict(records=records, paths=paths, gen_folder=gen_folder, rank_folder=rank_folder,
                feature_folder=feature_folder, rank_meta=rank_meta, splits=splits,
                grid=grid, grid_name=grid_name, common=common, feature_inputs=feature_inputs)


def training_config(prepared, record):
    return dict(prepared['common'], selection=record,
                feature_files=prepared['feature_inputs'][record['prf_id']])


def validate_existing(folder, config, num_features, splits):
    """Reject stale, partial, mismatched or corrupt saved outputs before skipping."""
    try:
        with open(folder / 'model.json') as handle:
            payload = json.load(handle)
        if payload['training_config'] != config:
            raise ValueError('saved configuration or source identities differ')
        if payload['artifacts']['parameters_sha256'] != sha256_file(folder / 'parameters.npz'):
            raise ValueError('parameter checksum differs (possibly interrupted write)')
        with np.load(folder / 'parameters.npz', allow_pickle=False) as params:
            for key in ('weights', 'feature_mean', 'feature_std', 'channel_indices'):
                if params[key].shape != (num_features,) or not np.all(np.isfinite(params[key])):
                    raise ValueError(f'invalid {key}')
            if np.any(params['feature_std'] <= 0):
                raise ValueError('nonpositive feature std')
            if not np.array_equal(params['channel_indices'], np.arange(num_features)):
                raise ValueError('channel index mismatch')
            for name, expected in zip(('train_ids', 'validation_ids', 'nested_ids'), splits):
                if not np.array_equal(params[name], expected):
                    raise ValueError(f'{name} mismatch')
            for key in ('intercept', 'nested_sse', 'best_lambda', 'lambda_candidates'):
                if not np.all(np.isfinite(params[key])):
                    raise ValueError(f'nonfinite {key}')
            index = int(params['best_lambda_index'])
            if not 0 <= index < len(params['lambda_candidates']):
                raise ValueError('invalid best lambda index')
            if params['best_lambda'] != params['lambda_candidates'][index]:
                raise ValueError('best lambda does not match its index')
    except (OSError, KeyError, ValueError, TypeError, EOFError) as error:
        raise ValueError(f'Cannot skip {folder}: {error}. Use a different --save_root '
                         'or explicitly --overwrite after checking the mismatch.') from error


def existing_action(args, prepared, record):
    folder = output_folder(args, record)
    if not folder.exists():
        return 'train'
    if args.overwrite:
        return 'overwrite'
    if args.skip_existing:
        validate_existing(folder, training_config(prepared, record),
                          prepared['rank_meta']['num_features'], prepared['splits'])
        return 'skip'
    raise FileExistsError(f'Output already exists: {folder}. Use --skip_existing or --overwrite.')


def save_result(args, prepared, record, result, runtime):
    weights, channels, best_index, loss, lambdas, std, mean = result
    weights = weights.detach().cpu().numpy()[:, 0]
    for name, values in (('weights', weights), ('mean', mean), ('std', std), ('loss', loss)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Nonfinite fitted {name}; refusing to save voxel {record["voxel_id"]}')
    if np.any(std <= 0):
        raise ValueError('Nonpositive fitted feature standard deviation')
    folder = output_folder(args, record)
    folder.parent.mkdir(parents=True, exist_ok=True)
    # Stage away from the final output. Publish model.json last as the commit
    # marker; its checksum detects an interrupted replacement of the NPZ/JSON pair.
    with tempfile.TemporaryDirectory(prefix=f'.{folder.name}.staging-', dir=folder.parent) as stage:
        train_ids, val_ids, nest_ids = prepared['splits']
        json_path, npz_path = save_online_ranking_model(
            stage, subject_id=args.subject_id, region=record['region'],
            voxel_id=record['voxel_id'], voxel_index_in_region=record['index_in_region'],
            voxel_rank=record['voxel_rank'], voxel_r2=record['voxel_r2'],
            generation_model_name=args.gen_model, model_name=args.rank_model,
            split_id=args.split_id, metadata=prepared['rank_meta'],
            layer_sizes=getattr(LayerSize, f'{args.rank_model}_layer_size'),
            prf_idx=record['prf_id'], prf_grid_name=prepared['grid_name'],
            prf_params=prepared['grid'][record['prf_id']], weights=weights[:-1],
            intercept=weights[-1], feature_mean=mean, feature_std=std,
            channel_indices=channels, lambda_candidates=lambdas,
            best_lambda_index=best_index, nested_sse=loss,
            train_ids=train_ids, val_ids=val_ids, nest_ids=nest_ids,
            features_model_folder=prepared['feature_folder'],
            ranking_model_path=prepared['rank_folder'], data_splits_path=prepared['paths']['splits'])
        with open(json_path) as handle:
            payload = json.load(handle)
        payload['training_mode'] = 'standalone'
        payload['ranking_encoder']['online_voxel_readout_fit'] = False
        payload['training_config'] = training_config(prepared, record)
        payload['runtime'] = runtime
        payload['artifacts']['parameters_sha256'] = sha256_file(npz_path)
        with open(json_path, 'w') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        validate_existing(Path(stage), payload['training_config'],
                          prepared['rank_meta']['num_features'], prepared['splits'])
        folder.mkdir(exist_ok=True)
        os.replace(npz_path, folder / 'parameters.npz')
        os.replace(json_path, folder / 'model.json')


def run(args):
    prepared = prepare_inputs(args)
    # Preflight every output before doing any training, including dry runs.
    actions = [existing_action(args, prepared, row) for row in prepared['records']]
    for row, action in zip(prepared['records'], actions):
        print(f'{action}: S{args.subject_id} {row["region"]} rank={row["voxel_rank"]} '
              f'voxel={row["voxel_id"]} pRF={row["prf_id"]} R²={row["voxel_r2"]:.6g} '
              f'-> {output_folder(args, row)}', flush=True)
    if args.dry_run:
        print('Dry run passed: inputs, ordering, split indices and feature headers checked. '
              'No training, CUDA initialization or writes. Full feature values are checked during fitting.')
        return
    if all(action == 'skip' for action in actions):
        print(f'All {len(actions)} matching artifacts already complete.')
        return
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is unavailable; no CPU fallback')
    runtime = dict(device=device, torch_version=torch.__version__,
                   numpy_version=np.__version__, cuda_version=torch.version.cuda,
                   gpu_name=torch.cuda.get_device_name(0) if device == 'cuda' else None)
    print(f'Runtime: {runtime}', flush=True)
    lock_dir = args.save_root / '.locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    for row in prepared['records']:
        folder = output_folder(args, row)
        lock_id = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()
        # Keep lock files: removing one while a process holds it creates a race.
        with open(lock_dir / f'{lock_id}.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'Another trainer is writing {folder}') from error
            if existing_action(args, prepared, row) == 'skip':
                print(f'Skipping verified artifact: {folder}', flush=True)
                continue
            started = time.monotonic()
            with h5py.File(prepared['paths']['betas'], 'r') as handle:
                # One actual big-mask column, not the entire subject on host/GPU.
                responses = handle['betas'][:, row['voxel_id']][:, None]
            train_ids, val_ids, nest_ids = prepared['splits']
            if not np.all(np.isfinite(responses[np.concatenate((train_ids, nest_ids))])):
                raise ValueError(f'Nonfinite fitting responses for voxel {row["voxel_id"]}')
            with torch.no_grad():
                result = rank_model_fitting(
                    0, row['prf_id'], responses, train_ids, val_ids, nest_ids,
                    prepared['feature_folder'], prepared['rank_meta']['layers'], device)
            save_result(args, prepared, row, result,
                        dict(runtime, fit_elapsed_seconds=time.monotonic() - started))
            del result, responses
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'Saved {folder} ({time.monotonic() - started:.1f}s)', flush=True)
    print(f'Complete: {len(prepared["records"])} requested ranking readouts.', flush=True)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
