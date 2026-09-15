# Seed selection and recording

The concatenated generation runner accepts three modes:

```bash
# Fresh random seed sequence (default).
--seed_mode random

# Reproducible image-seed sequence, without changing global Python RNG state.
--seed_mode fixed --seed_generator_seed 42

# Replay attempts from a previous voxel.
--seed_mode file --seed_file /path/to/voxel/seeds.json

# Or replay only its fully valid seeds.
--seed_mode file --seed_file /path/to/voxel/seeds.json --seed_file_key valid_seeds
```

A seed file may also be a bare JSON array, e.g. `[101, 202, 303]`.
Object files support `all_seeds_attempted` (default), `valid_seeds`, or `seed_set`
via `--seed_file_key`. Entries must be unique integers in `[0, 2**32-1]`.
File order is preserved. Invalid/empty files fail before generation starts.

Each run creates one ordered sequence, reused from the beginning for each voxel.
Random/fixed modes generate enough seeds for `--max_generation_attempts` (default:
`num_seeds + max(10, ceil(num_seeds * 0.1))`). `--num_seeds` remains the requested
number of fully valid candidates, not the number of attempts. Generation stops
when it reaches that number, the attempt limit, or the end of a loaded list.
File mode never silently generates extra seeds; shortfalls are saved as partial
results. Include replacement seeds in the file if needed. Loaded seeds can fail
under different models/constraints, even if previously valid.

Fixed mode repeats the sequence for the same generator seed. Increasing the
attempt budget preserves its prefix. This does not guarantee bit-identical
images across hardware/software or otherwise seed all model-training randomness.
For matched seeds across the DINO and ADV Slurm tasks, use the same fixed seed
or input file; random mode gives each task its own fresh sequence.

Outputs include a seed-plan subfolder to separate different seed configurations:

```text
<save_root>/sub-XX/<models>/<settings>/seed-<mode>-<plan_id>/
├── run_config.json                    # Includes the seed plan
├── generation_timing.json             # Total generation time and per-voxel summaries
└── roi-.../voxel-rank-...__id-.../
    ├── seeds.json
    ├── ranking.json                   # Links to seeds.json
    ├── ranking_model/
    ├── images/
    └── unconstrained_images/          # When applicable
```

`seeds.json` contains `seed_set` (the full planned/loaded sequence),
`all_seeds_attempted` (ordered actual attempts), and `valid_seeds` (ordered seeds
passing image checks and finite generation/ranking scores, before top-N selection).
It also records mode, fixed generator seed when applicable, source file absolute
path and SHA-256 when applicable, selected file key, attempt limits, and status.
The record is atomically updated before each attempt and after each scoring batch.
It is also updated after every pipeline call (including calls raising an error).
`seeds.json` contains `generation_timing.per_seed`, with each seed's
`generation_seconds`, attempt number, whether the pipeline returned normally,
and whether it is currently in the fully valid list. The timing section also
contains total `generation_seconds`, `generation_calls`, and average
`seconds_per_generation`. A returned pipeline call can still fail subsequent
pixel/contrast/luminance/score checks. Pending/unscored seeds are not valid yet.

Timing measures wall-clock time inside each diffusion pipeline call, with CUDA
synchronization around the call. It includes guidance, decoding and in-pipeline
constraints, but excludes model loading, ranking-model training, scoring,
external candidate validation and saving. Rejected calls contribute to totals.
The experiment's `generation_timing.json` is checkpointed alongside seed records
and aggregates all voxels in the current process invocation, with links to their
seed records. Separate Slurm array tasks have separate timing files; these totals
are not full job elapsed time. `--generation_timing_file` optionally writes an
additional copy at an explicit location. Timing JSON files also contain a
`hardware` object: GPU model, process-visible CUDA index, UUID when exposed by
PyTorch, total VRAM, compute capability, hostname, CUDA visibility, Slurm job/task
IDs, and PyTorch/CUDA versions. The index is local to the process's visible GPUs,
not necessarily the physical node index. CPU runs have null GPU fields.
Hardware metadata is not added to `seeds.json`.
A forcibly killed in-flight call cannot
have a completed duration recorded, although its attempted seed is recorded.

After an interrupted run, `in_progress` means some attempts may not have finished
or been scored. Recording does not automatically resume an interrupted run.
Reusing an identical plan/configuration reuses its output path; use a different
`--save_root` for independent reruns that must preserve previous outputs.

Both Slurm scripts expose `SEED_MODE`, `SEED_GENERATOR_SEED`, `SEED_FILE`, and
`SEED_FILE_KEY`. For example, from `/home/junruz/BrainDiVE`:

```bash
sbatch --export=ALL,SEED_MODE=fixed,SEED_GENERATOR_SEED=42 \
  script/test_img_gen_prf_concat_model_rank.job
```

Combine these exports with existing contrast/luminance settings. Omit file/fixed
arguments when using random mode; conflicting arguments are configuration errors.
