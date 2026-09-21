#!/usr/bin/env python3
"""Score original and adjusted BrainDiVE PNGs with both readouts; export six top-N sets."""

import argparse
import csv
import fcntl
import heapq
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from dual_scope_contrast_utils import DualScopeContrastConvergenceError
from joint_luma_contrast_utils import joint_result_to_jsonable
from postprocess_ranked_utils import (
    FINAL_ACCEPTANCE_DOMAIN, UNADJUSTED_ACCEPTANCE_DOMAIN, FinalPNGValidationError,
    LayerSize, adjust_pixels, build_encoder, build_kernels, file_identity,
    generation_readout, get_prf_grid, load_concat_metadata, load_generation,
    load_rank_readout, prepare_unadjusted_pixels, read_pixels, score_pixels, settings_from_generation,
    sha256_file, validate_concat_layout,
)

PROJECT = Path('/ocean/projects/soc250009p/jzhao7')
MODES = ('unadjusted', 'full_only', 'full_and_prf')
SCHEMA = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--sample', action='store_true', help='Bounded sample, written beneath _validation')
    scope.add_argument('--full-run', action='store_true', help='Process every image in selected voxel groups')
    parser.add_argument('--input-root', type=Path, default=PROJECT / 'data/output/BrainDiVE/pRF_concat_model_ranked')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'data/output/BrainDiVE/ICLR2027_top30')
    parser.add_argument('--model-root', type=Path, default=PROJECT / 'data/model/concat/split_1_zscore')
    parser.add_argument('--ranking-root', type=Path, default=PROJECT / 'data/model/ranking')
    parser.add_argument('--subjects', nargs='+', type=int, choices=range(1, 9), default=[2, 5, 7])
    parser.add_argument('--rois', nargs='+', choices=['V1', 'hV4'], default=['V1', 'hV4'])
    parser.add_argument('--gen-models', nargs='+', choices=['ADV_RN50', 'DINO_RN50'],
                        default=['ADV_RN50', 'DINO_RN50'])
    parser.add_argument('--split', type=int, choices=[1, 2], default=1)
    parser.add_argument('--voxel-rank-range', nargs=2, type=int, metavar=('START', 'END'))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--sample-images', type=int, default=2, help='First N recorded images per sample voxel')
    parser.add_argument('--sample-voxels', type=int, default=1, help='First N voxels per subject/ROI/generation model')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--adjustment-device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='auto uses the scoring device; cuda batches both adjustment modes on GPU')
    parser.add_argument('--gpu-solver-max-iterations', type=int, default=100)
    parser.add_argument('--adv-checkpoint', type=Path, default=PROJECT / 'data/model/imagenet_l2_3_0.pt')
    parser.add_argument('--dino-checkpoint', type=Path,
                        default=Path(torch.hub.get_dir()) / 'checkpoints/dino_resnet50_pretrain.pth')
    parser.add_argument('--open-clip-checkpoint', type=Path, help='Local OPEN_CLIP RN50 yfcc15m checkpoint')
    parser.add_argument('--resume', action='store_true', help='Verify and skip matching completed mode outputs')
    parser.add_argument('--dry-run', action='store_true', help='Read metadata/checkpoints only; no encoder or output writes')
    args = parser.parse_args(argv)
    for field in ('top_n', 'sample_images', 'sample_voxels', 'batch_size', 'threads', 'gpu_solver_max_iterations'):
        if getattr(args, field) <= 0:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    if args.voxel_rank_range and not 1 <= args.voxel_rank_range[0] <= args.voxel_rank_range[1]:
        parser.error('Invalid inclusive voxel rank range')
    for field in ('subjects', 'rois', 'gen_models', 'modes'):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f'Duplicate --{field.replace("_", "-")} values')
    return args


def json_dump(path, value):
    with open(path, 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def canonical_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def backbone_paths(args):
    clip_path = args.open_clip_checkpoint
    if clip_path is None:
        # Resolve the cached main revision explicitly; do not download or select
        # an arbitrary revision when multiple snapshots are present.
        repo = PROJECT / 'data/huggingface/hub/models--timm--resnet50_clip.yfcc15m'
        revision = (repo / 'refs/main').read_text().strip()
        snapshot = repo / 'snapshots' / revision
        clip_path = snapshot / 'open_clip_model.safetensors'
        if not clip_path.is_file():
            clip_path = snapshot / 'open_clip_pytorch_model.bin'
    paths = {'ADV_RN50': args.adv_checkpoint, 'DINO_RN50': args.dino_checkpoint,
             'OPEN_CLIP_RN50': clip_path}
    # Preserve the snapshot filename extension for the safetensors loader.
    paths = {name: path.absolute() for name, path in paths.items()
             if name in args.gen_models or name == 'OPEN_CLIP_RN50'}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Local encoder checkpoint missing: {path}')
    return paths


def discover(args):
    """Use metadata as the image inventory; reject ambiguous runs instead of merging them."""
    groups = []
    for subject in args.subjects:
        for model in args.gen_models:
            for roi in args.rois:
                pattern = (f'sub-{subject:02d}/gen-{model}-concat/ranking-disabled/method-luma_rgb_clip/'
                           f'split-{args.split:02d}*/*/roi-{roi}/voxel-*/generation.json')
                paths = sorted(args.input_root.glob(pattern), key=lambda p: (p.parent.name, str(p)))
                selected, seen = [], set()
                for path in paths:
                    match = re.fullmatch(r'voxel-rank-(\d+)__id-(\d+)', path.parent.name)
                    if not match:
                        raise ValueError(f'Unrecognized voxel folder: {path.parent}')
                    rank, voxel = map(int, match.groups())
                    if args.voxel_rank_range and not args.voxel_rank_range[0] <= rank <= args.voxel_rank_range[1]:
                        continue
                    if voxel in seen:
                        raise ValueError(f'Multiple source runs for S{subject}/{model}/{roi}/voxel {voxel}; narrow --input-root')
                    seen.add(voxel)
                    selected.append((rank, voxel, path))
                if args.sample:
                    selected = selected[:args.sample_voxels]
                if not selected:
                    raise ValueError(f'No source groups for S{subject}/{model}/{roi}')
                for rank, voxel, path in selected:
                    d = json.loads(path.read_text())
                    if (d['subject'], d['generation_model']['name'], d['region'], d['split'],
                            d['voxel_rank'], d['voxel_id']) != (subject, model, roi, args.split, rank, voxel):
                        raise ValueError(f'Generation metadata disagrees with source path: {path}')
                    settings_from_generation(d)
                    records = d['results'][:args.sample_images] if args.sample else d['results']
                    if not records or len({r['seed'] for r in records}) != len(records):
                        raise ValueError(f'Empty or duplicate source seed inventory: {path}')
                    images = []
                    for record in records:
                        image_path = (path.parent / record['raw_image_file']).resolve()
                        if not image_path.is_relative_to(path.parent.resolve()):
                            raise ValueError(f'Source image escapes voxel folder: {image_path}')
                        images.append(dict(seed=int(record['seed']), source_path=str(image_path)))
                    d.pop('results')
                    groups.append(dict(path=path, generation=d, images=images))
    return groups


def output_folder(args, d):
    root = args.output_root
    if args.sample:
        root = root / '_validation' / f'sample-{args.sample_images}-per-voxel-v{SCHEMA}'
    return (root / f'sub-{d["subject"]:02d}' / f'gen-{d["generation_model"]["name"]}' /
            f'split-{d["split"]:02d}' / f'roi-{d["region"]}' /
            f'voxel-rank-{d["voxel_rank"]:02d}__id-{d["voxel_id"]:06d}')


def prepare_group(args, group, backbones, code_hashes):
    d = group['generation']
    model = d['generation_model']['name']
    gen_folder = args.model_root / f'S{d["subject"]}' / f'{model}_set{d["split"]}'
    rank_folder = (args.ranking_root / f'S{d["subject"]}' / f'gen-{model}__rank-OPEN_CLIP_RN50' /
                   f'split-{d["split"]:02d}' / f'roi-{d["region"]}' / group['path'].parent.name)
    gen_meta = load_concat_metadata(gen_folder, model)
    sizes = getattr(LayerSize, f'{model}_layer_size')
    validate_concat_layout(gen_meta, sizes, 'Generation')
    if d['prf_grid_name'] != 'default-log-polar' or d['prf_idx'] not in gen_meta['prf_ids']:
        raise ValueError('Unsupported generation pRF grid or ID')
    grid, _ = get_prf_grid()
    if not np.allclose(grid[d['prf_idx']], [d['prf_x'], d['prf_y'], d['prf_sigma']], rtol=0, atol=1e-7):
        raise ValueError('Generation pRF coordinates disagree with the current grid')
    prfs = np.load(gen_folder / 'best_prf_idx.npy', mmap_mode='r')
    if int(np.asarray(prfs[d['voxel_id']]).item()) != d['prf_idx']:
        raise ValueError('Generation checkpoint pRF changed since image generation')
    if d['contrast_prf_native_size'] != sizes[d['contrast_prf_layer']][-2:]:
        raise ValueError('Contrast pRF native size mismatch')
    if d['contrast_prf_layer'] != max(gen_meta['layers'], key=lambda layer: sizes[layer][-1]):
        raise ValueError('Contrast pRF is not from the highest-resolution generation layer')
    rank_readout, rank_meta = load_rank_readout(rank_folder, d, gen_meta)
    sources = []
    for record in group['images']:
        try:
            identity = file_identity(record['source_path'])
        except FileNotFoundError:
            identity = dict(path=record['source_path'], missing=True)
        sources.append(dict(seed=record['seed'], **identity))
    files = [gen_folder / name for name in ('concat_metadata.json', 'best_weights.pkl',
             'best_features_m.npy', 'best_features_s.npy', 'best_prf_idx.npy')]
    config = dict(schema_version=SCHEMA, source_metadata=str(group['path'].resolve()),
                  source_metadata_sha256=sha256_file(group['path']), source_images=sources,
                  generation_checkpoints=[file_identity(p) for p in files],
                  ranking_metadata_sha256=sha256_file(rank_folder / 'model.json'),
                  ranking_parameters_sha256=sha256_file(rank_folder / 'parameters.npz'),
                  ranking_folder=str(rank_folder.resolve()),
                  backbones={n: file_identity(backbones[n]) for n in (model, 'OPEN_CLIP_RN50')},
                  implementation=code_hashes, settings=settings_from_generation(d), top_n=args.top_n,
                  sample=args.sample, tie_break='seed_ascending',
                  adjustment_device=args.adjustment_device,
                  adjustment_backend='torch_cuda_batched_luma_lm_v1' if args.adjustment_device == 'cuda' else 'reference_cpu',
                  gpu_solver_max_iterations=args.gpu_solver_max_iterations if args.adjustment_device == 'cuda' else None,
                  subject=d['subject'], region=d['region'], voxel_id=d['voxel_id'],
                  generation_model=model, ranking_model='OPEN_CLIP_RN50', prf_id=d['prf_idx'])
    return dict(config=config, gen_folder=gen_folder, gen_meta=gen_meta,
                rank_meta=rank_meta, rank_readout=rank_readout)


def mode_config(base, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown image mode: {mode}')
    adjusted = mode != 'unadjusted'
    preparation = (['final_adjustment', 'clip_rgb', 'quantize_uint8', 'validate_final_pixels']
                   if adjusted else ['preserve_original_pixels'])
    return dict(base, adjustment_mode=mode,
                settings=base['settings'] if adjusted else None,
                final_adjustment_applied=adjusted, final_target_validation_applied=adjusted,
                acceptance_domain=FINAL_ACCEPTANCE_DOMAIN if adjusted else UNADJUSTED_ACCEPTANCE_DOMAIN,
                postclip_mismatch_policy='reject_after_quantization' if adjusted else 'not_applicable',
                scores_computed_on='quantized_RGB_PNG_pixels' if adjusted else 'original_RGB_PNG_pixels',
                processing_order=['load_source_png', *preparation, 'score_generation_and_ranking',
                                  'select_top_n', 'export_scored_pixels'])


class TopPixels:
    """Keep the union of both top-N lists in memory, at most 2*N PNG arrays."""

    def __init__(self, count):
        self.count = count
        self.heaps = {'gen_score': [], 'rank_score': []}
        self.pixels = {}

    def add(self, row, pixels):
        for key, heap in self.heaps.items():
            item = (row[key], -row['seed'], row['seed'])
            if len(heap) < self.count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        kept = {item[2] for heap in self.heaps.values() for item in heap}
        if row['seed'] in kept:
            self.pixels[row['seed']] = pixels
        for seed in self.pixels.keys() - kept:
            del self.pixels[seed]


def check_completed(folder, config, resume):
    if not folder.exists():
        return False
    if not resume:
        raise FileExistsError(f'Output exists; use --resume to verify it: {folder}')
    marker = folder / 'summary.json'
    if not marker.is_file():
        raise ValueError(f'Incomplete output: {folder}; use a different --output-root')
    saved = json.loads(marker.read_text())
    if saved['config'] != config or saved['status'] != 'complete':
        raise ValueError(f'Output configuration changed: {folder}; use a different --output-root')
    for name, checksum in saved['output_sha256'].items():
        path = folder / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f'Incomplete or modified result: {path}')
    return True


def process_mode(stage, group, mode, config, weight, scorer, batch_size):
    started = time.monotonic()
    top = TopPixels(config['top_n'])
    rows, pending, failures = [], [], []
    diagnostics_path = stage / 'diagnostics.jsonl'

    def flush(handle):
        if not pending:
            return
        gen_scores, rank_scores = scorer([item[1] for item in pending])
        if len(gen_scores) != len(pending) or len(rank_scores) != len(pending):
            raise ValueError('Scorer returned the wrong number of predictions')
        for (source, pixels, diagnostics), gs, rs in zip(pending, gen_scores, rank_scores):
            if not math.isfinite(gs) or not math.isfinite(rs):
                failures.append(dict(**source, reason='nonfinite_model_score',
                                     gen_score=float(gs) if math.isfinite(gs) else None,
                                     rank_score=float(rs) if math.isfinite(rs) else None))
                continue
            import hashlib
            row = dict(**source, gen_score=float(gs), rank_score=float(rs),
                       pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                       **{f'png_{key}_byte': value for key, value in diagnostics['png_stats_byte'].items()})
            rows.append(row)
            top.add(row, pixels)
            handle.write(json.dumps(dict(**source, diagnostics=diagnostics), allow_nan=False) + '\n')
        pending.clear()

    def record_failure(source, error):
        failure = dict(**source, reason=error.reason, message=str(error))
        if error.result is not None:
            failure['solver'] = joint_result_to_jsonable(error.result)
        if hasattr(error, 'diagnostics'):
            failure['diagnostics'] = error.diagnostics
        failures.append(failure)

    with open(diagnostics_path, 'w') as handle:
        for start in range(0, len(group['images']), batch_size):
            loaded = []
            for source in group['images'][start:start + batch_size]:
                try:
                    loaded.append((source, read_pixels(source['source_path'])))
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    failures.append(dict(**source, reason='invalid_source_image', message=str(error)))
            if mode != 'unadjusted' and config.get('adjustment_device') == 'cuda':
                from postprocess_gpu_adjustment import adjust_pixels_cuda
                # Different image sizes can be decoded, so batch only like shapes.
                outcomes = [None] * len(loaded)
                shapes = {}
                for i, (_, pixels) in enumerate(loaded):
                    shapes.setdefault(pixels.shape, []).append(i)
                for ids in shapes.values():
                    results = adjust_pixels_cuda([loaded[i][1] for i in ids], mode, weight,
                                                 config['settings'],
                                                 max_iterations=config['gpu_solver_max_iterations'])
                    if len(results) != len(ids):
                        raise RuntimeError('GPU preparation returned the wrong batch size')
                    for i, result in zip(ids, results):
                        outcomes[i] = result
            else:
                outcomes = []
                for source, pixels in loaded:
                    try:
                        outcomes.append(prepare_unadjusted_pixels(pixels, weight) if mode == 'unadjusted'
                                        else adjust_pixels(pixels, mode, weight, config['settings']))
                    except DualScopeContrastConvergenceError as error:
                        outcomes.append(error)
            for (source, _), outcome in zip(loaded, outcomes):
                if isinstance(outcome, DualScopeContrastConvergenceError):
                    record_failure(source, outcome)
                    continue
                prepared_pixels, diagnostics = outcome
                # Only accepted final uint8 pixels reach either scorer.
                pending.append((source, prepared_pixels, diagnostics))
                if len(pending) >= batch_size:
                    flush(handle)
            done = min(start + batch_size, len(group['images']))
            if done == len(group['images']) or done // 100 > start // 100:
                print(f'  {mode}: prepared {done}/{len(group["images"])}', flush=True)
        flush(handle)

    selections = {}
    for score, rank_key, folder_name in (('gen_score', 'gen_rank', 'top30_generation'),
                                          ('rank_score', 'rank_model_rank', 'top30_ranking')):
        # Names remain top30_* for the requested default; custom N is explicit.
        folder_name = folder_name.replace('30', str(config['top_n']))
        target = stage / folder_name
        target.mkdir()
        ordered = sorted(rows, key=lambda r: (-r[score], r['seed']))
        selections[rank_key] = []
        for rank, row in enumerate(ordered, 1):
            row[rank_key] = rank
            if rank <= config['top_n']:
                relative = f'{folder_name}/rank-{rank:03d}__seed-{row["seed"]}.png'
                Image.fromarray(top.pixels[row['seed']]).save(stage / relative)
                selections[rank_key].append(dict(seed=row['seed'], score=row[score], image=relative))
    columns = ['seed', 'source_path', 'gen_score', 'rank_score', 'gen_rank', 'rank_model_rank',
               'pixels_sha256', 'png_global_mean_byte', 'png_global_rms_byte', 'png_prf_mean_byte', 'png_prf_rms_byte']
    with open(stage / 'scores.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(stage / 'failures.jsonl', 'w') as handle:
        for failure in failures:
            handle.write(json.dumps(failure, allow_nan=False) + '\n')
    output_hashes = {str(p.relative_to(stage)): sha256_file(p) for p in stage.rglob('*') if p.is_file()}
    summary = dict(status='complete', config=config, input_count=len(group['images']), valid_count=len(rows),
                   skipped_count=len(failures), top_n_requested=config['top_n'],
                   top_n_returned=min(config['top_n'], len(rows)),
                   shortfall=max(0, config['top_n'] - len(rows)), selections=selections,
                   elapsed_seconds=time.monotonic() - started, output_sha256=output_hashes)
    if len(rows) + len(failures) != len(group['images']):
        raise RuntimeError('Candidate accounting mismatch')
    json_dump(stage / 'summary.json', summary)
    return summary


def run(args):
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.adjustment_device == 'auto':
        args.adjustment_device = device
    if args.adjustment_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA adjustment requested but unavailable')
    backbones = backbone_paths(args)
    source_dir = Path(__file__).resolve().parent
    code_hashes = {name: sha256_file(source_dir / name) for name in (
        'postprocess_ranked_images.py', 'postprocess_ranked_utils.py', 'luma_rgb_clip_utils.py',
        'joint_luma_contrast_utils.py', 'dual_scope_contrast_utils.py', 'prf_utils.py',
        'ranking_model_training_utils.py', 'postprocess_gpu_adjustment.py')}
    groups = discover(args)
    print(f'{"SAMPLE" if args.sample else "FULL"}: {len(groups)} voxel/model groups, '
          f'{sum(len(g["images"]) for g in groups)} source PNGs, modes={args.modes}, device={device}, '
          f'adjustment_device={args.adjustment_device}', flush=True)
    encoders, cached_generation, cached_key = {}, None, None
    total_valid = total_skipped = 0
    for group in groups:
        prepared = prepare_group(args, group, backbones, code_hashes)
        d = group['generation']
        root = output_folder(args, d)
        print(f'S{d["subject"]} {d["region"]} {d["generation_model"]["name"]} voxel {d["voxel_id"]}: {len(group["images"])} inputs', flush=True)
        for mode in args.modes:
            config = mode_config(prepared['config'], mode)
            folder = root / mode
            if check_completed(folder, config, args.resume):
                print(f'  verified existing {folder}', flush=True)
                continue
            if args.dry_run:
                print(f'  preflight OK -> {folder}', flush=True)
                continue
            root.mkdir(parents=True, exist_ok=True)
            # One lock per voxel/mode. Staging is never mistaken for completion.
            with open(root / f'.{mode}.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if check_completed(folder, config, args.resume):
                    continue
                model = d['generation_model']['name']
                for name in (model, 'OPEN_CLIP_RN50'):
                    if name not in encoders:
                        print(f'  loading local {name} encoder', flush=True)
                        encoders[name] = build_encoder(name, backbones[name], device)
                key = str(prepared['gen_folder'])
                if cached_key != key:
                    cached_generation = None
                    cached_generation = load_generation(prepared['gen_folder'])
                    cached_key = key
                gen_readout = generation_readout(cached_generation, d['voxel_id'], prepared['gen_meta']['num_features'])
                gen_layers, rank_layers = prepared['gen_meta']['layers'], prepared['rank_meta']['layers']
                gen_kernels = build_kernels(gen_layers, model, d['prf_idx'], device)
                rank_kernels = build_kernels(rank_layers, 'OPEN_CLIP_RN50', d['prf_idx'], device)
                weight = gen_kernels[d['contrast_prf_layer']].cpu()

                def scorer(pixels):
                    return (score_pixels(pixels, model, encoders[model], gen_layers, gen_kernels, gen_readout, device),
                            score_pixels(pixels, 'OPEN_CLIP_RN50', encoders['OPEN_CLIP_RN50'], rank_layers,
                                         rank_kernels, prepared['rank_readout'], device))

                with tempfile.TemporaryDirectory(prefix=f'.{mode}.staging-', dir=root) as temporary:
                    stage = Path(temporary) / mode
                    stage.mkdir()
                    summary = process_mode(stage, group, mode, config, weight, scorer, args.batch_size)
                    summary['runtime'] = dict(device=device, torch_version=torch.__version__, threads=args.threads,
                                              slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                              adjustment_device=args.adjustment_device,
                                              adjustment_backend=config['adjustment_backend'])
                    json_dump(stage / 'summary.json', summary)
                    os.rename(stage, folder)
                total_valid += summary['valid_count']
                total_skipped += summary['skipped_count']
                print(f'  {mode}: {summary["valid_count"]} valid, {summary["skipped_count"]} skipped; '
                      f'saved {summary["top_n_returned"]} per ranking ({summary["elapsed_seconds"]:.1f}s)', flush=True)
    print(f'Done: {total_valid} valid image versions, {total_skipped} skipped in this invocation.', flush=True)


if __name__ == '__main__':
    run(parse_args())
