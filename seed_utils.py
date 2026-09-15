"""Independent diffusion-seed plans and atomic per-voxel audit records."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import tempfile


def build_seed_plan(mode, num_seeds, max_attempts=None, seed_file=None,
                    generator_seed=None, file_key='all_seeds_attempted'):
    """Use one ordered plan per run; each voxel starts at its first seed."""
    limit = max_attempts if max_attempts is not None else num_seeds + max(10, math.ceil(num_seeds * .1))
    if num_seeds < 1 or limit < num_seeds or limit > 2**32:
        raise ValueError('Seed counts must satisfy 1 <= num_seeds <= max attempts <= 2**32')
    if mode not in ('random', 'file', 'fixed'):
        raise ValueError(f'Unknown seed mode: {mode}')
    if mode != 'file' and seed_file is not None:
        raise ValueError('--seed_file requires --seed_mode file')
    if mode != 'fixed' and generator_seed is not None:
        raise ValueError('--seed_generator_seed requires --seed_mode fixed')
    source = None
    digest = None
    if mode == 'file':
        if seed_file is None:
            raise ValueError('--seed_mode file requires --seed_file')
        source = str(Path(seed_file).resolve())
        raw = Path(source).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        seeds = data if isinstance(data, list) else data[file_key]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError('Seed file must contain a nonempty JSON seed list')
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError('Every image seed must be an integer in [0, 2**32-1]')
        if len(set(seeds)) != len(seeds):
            raise ValueError('Seed file contains duplicate seeds')
    else:
        if mode == 'fixed' and generator_seed is None:
            raise ValueError('--seed_mode fixed requires --seed_generator_seed')
        rng = random.Random(generator_seed) if mode == 'fixed' else random.SystemRandom()
        # Draw sequentially so increasing the attempt budget preserves the prefix.
        seeds, seen = [], set()
        while len(seeds) < limit:
            seed = rng.randrange(2**32)
            if seed not in seen:
                seeds.append(seed)
                seen.add(seed)
    plan = {
        'mode': mode, 'seed_generator_seed': generator_seed,
        'seed_file': source, 'seed_file_sha256': digest,
        'seed_file_key': file_key if mode == 'file' else None,
        'seed_set': seeds, 'requested_valid_seeds': num_seeds,
        'configured_max_attempts': limit,
        'effective_max_attempts': min(limit, len(seeds)),
        'sequence_scope': 'same_ordered_sequence_for_each_voxel',
    }
    plan['plan_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    return plan


def summarize_generation_times(records):
    seconds = sum(record['generation_seconds'] for record in records)
    return {
        'generation_seconds': seconds,
        'generation_calls': len(records),
        'seconds_per_generation': seconds / len(records) if records else None,
    }


GENERATION_TIMING_SCOPE = (
    'Wall-clock seconds inside diffusion pipeline calls, including guidance, '
    'decoding and in-pipeline constraints; CUDA synchronized before/after each call. '
    'Includes failed calls; excludes model loading, ranking-model training, scoring, '
    'external candidate validation and file saving.'
)


def collect_device_metadata(device, torch_module):
    """Describe the actual process-visible generation device, not every node GPU."""
    metadata = {
        'device_type': device.type,
        'hostname': socket.gethostname(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'torch_version': str(torch_module.__version__),
        'cuda_version': torch_module.version.cuda,
        'cuda_device_index': None,
        'gpu_name': None, 'gpu_uuid': None,
        'total_memory_bytes': None, 'total_memory_gib': None,
        'compute_capability': None,
    }
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(index)
        uuid = getattr(properties, 'uuid', None)
        metadata.update(
            cuda_device_index=index, gpu_name=properties.name,
            gpu_uuid=str(uuid) if uuid is not None else None,
            total_memory_bytes=int(properties.total_memory),
            total_memory_gib=properties.total_memory / 2**30,
            compute_capability=f'{properties.major}.{properties.minor}',
        )
    return metadata


def save_seed_record(path, plan, attempted, valid, status='in_progress', generation_times=None,
                     ranking_enabled=True):
    """Checkpoint before each attempt and after scoring; do not expose partial JSON."""
    record = dict(plan, all_seeds_attempted=list(attempted), valid_seeds=list(valid),
                  status=status,
                  ranking_enabled=ranking_enabled,
                  valid_seed_definition=('passed_image_checks_and_finite_generation_and_ranking_scores'
                                         if ranking_enabled else 'passed_image_and_constraint_checks_without_post_generation_scoring'))
    if generation_times is not None:
        valid_set = set(valid)
        record['generation_timing'] = {
            **summarize_generation_times(generation_times),
            'scope': GENERATION_TIMING_SCOPE,
            'per_seed': [dict(item, valid_seed=item['seed'] in valid_set)
                         for item in generation_times],
        }
    save_json_atomic(path, record)


def save_json_atomic(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
