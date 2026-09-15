"""Atomic, locked timing snapshots shared by workers within one ROI."""

import fcntl
import json
import os
from pathlib import Path

from seed_utils import GENERATION_TIMING_SCOPE, save_json_atomic


def merge_roi_generation_timing(path, region, updates):
    """Replace cumulative voxel snapshots without dropping other workers' voxels."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock a stable separate inode, since saving atomically replaces the JSON inode.
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads(path.read_text()) if path.exists() else {}
        rows = {}
        for item in old.get('per_voxel', []):
            row = dict(item)
            row.setdefault('hardware', old.get('hardware'))
            rows[os.path.realpath(row['seed_record'])] = row
        for item in updates:
            rows[os.path.realpath(item['seed_record'])] = dict(item)
        values = [rows[key] for key in sorted(rows)]
        if any(row['region'] != region for row in values):
            raise ValueError('ROI timing file cannot contain multiple regions')
        seconds = sum(row['generation_seconds'] for row in values)
        calls = sum(row['generation_calls'] for row in values)
        hardware = values[0].get('hardware') if values else None
        if any(row.get('hardware') != hardware for row in values):
            hardware = None
        payload = {
            'schema_version': 2, 'region': region,
            'hardware': hardware,
            'hardware_note': 'Per-voxel hardware is authoritative; top-level null means mixed or unknown hardware.',
            'scope': GENERATION_TIMING_SCOPE,
            'aggregation': 'latest_snapshot_per_voxel_in_this_roi',
            'generation_seconds': seconds,
            'generation_calls': calls,
            'seconds_per_generation': seconds / calls if calls else None,
            'per_voxel': values,
        }
        save_json_atomic(path, payload)
        return payload
