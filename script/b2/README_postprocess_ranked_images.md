# Final adjustment and top-30 export

`postprocess_ranked_images.py` processes the existing pre-adjustment RGB PNGs.
It evaluates three independent branches: `unadjusted`, `full_only`, and
`full_and_prf`. Each branch is scored with both the generation readout and its
saved OPEN_CLIP ranking readout, and gets its own two top-N selections. The
adjusted branches use the recorded `luma_rgb_clip` controls. No diffusion generation or readout fitting
is performed. Encoder weights are loaded exclusively from local checkpoints.

Defaults: subjects 2/5/7, V1/hV4, ADV/DINO, split 1, top 30 per voxel/model/mode.
`unadjusted` preserves the original saved RGB PNG pixels, skips final projection
and luminance/contrast target checks, and records the measured statistics. It
requires a readable RGB PNG and finite scores from both models. It is evaluated
independently even when an adjusted version fails. These originals can already
reflect constraints applied during diffusion; “unadjusted” means no final
adjustment. Use `--modes unadjusted` to run just this branch.

`full_only` matches whole-image luminance and RMS; `full_and_prf` also matches
the generation voxel's soft-pRF-weighted statistics. The source metadata sets
mean 116, RMS 57.48, tolerance 3 (0–255 units), 50 full-image match iterations,
and 300 dual-scope solver evaluations. Original chroma is restored then RGB is
clipped and quantized to uint8. Acceptance checks Rec.709 luminance statistics
on those final RGB pixels: full-image mean/RMS for `full_only`, and all four
full-image/pRF statistics for `full_and_prf`. Every active statistic must be
within its recorded tolerance (3 in the current experiment). An image that
fails is skipped with reason `final_png_target_mismatch`, including measured
statistics, targets, tolerances, and errors in `failures.jsonl`.

The existing projection solver and its convergence checks are unchanged; this
adds final-PNG acceptance rather than refitting the solver objective to clipped
RGB. Thus some projections that previously passed can now be rejected. Input
quantization is unavoidable: the original float decodes were not saved.

Generation scoring uses the original concatenated encoding weights, per-voxel
feature normalization, and generation pRF. Ranking uses `parameters.npz` under
`data/model/ranking`, verifies the checksum and subject/voxel/split/pRF/layer
identity, and reuses the readout trained at the generation pRF. Both scorers
use bilinear antialiased 224×224 preprocessing and native-layer Gaussian
pooling. Scores are predicted voxel responses, descending; ties use ascending
seed. Both scores must be finite. The order for each adjusted image is:
projection → RGB clipping → uint8 quantization → final-pixel target validation
→ both model scores → selection → export of the exact scored pixels. Unadjusted
images go directly from their original decoded uint8 pixels to scoring and
selection. No transformations are applied after scoring. Thus there are six
independent top-N sets per voxel/model group by default.

## Bounded validation

From `/ocean/projects/soc250009p/jzhao7/code/BrainDiVE`:

```bash
# Read-only preflight: 12 voxel/model groups, 24 source PNGs.
bash script/b2/postprocess_ranked_images.job --dry-run

# Small CPU sample, using two source images from one voxel per group.
bash script/b2/postprocess_ranked_images.job --device cpu --threads 2 --batch-size 2

# Verify completed outputs and skip them on restart.
bash script/b2/postprocess_ranked_images.job --device cpu --threads 2 --batch-size 2 --resume
```

The sample writes only beneath
`data/output/BrainDiVE/ICLR2027_top30/_validation/sample-2-per-voxel-v3/`.
Earlier samples under `sample-2-per-voxel` and `sample-2-per-voxel-final-rgb`
are retained separately. They predate the unadjusted branch (the first also
used pre-clipping acceptance). Resume rejects incompatible configuration or
implementation metadata.
Use `--sample-images N`, `--sample-voxels N`, `--subjects`, `--rois`,
`--gen-models`, or `--voxel-rank-range START END` to change the sample.
These limits take the first recorded images and the first selected voxel ranks.
They do not retry or replace failed images to reach a successful-image quota.
The script requires an explicit `--sample` or `--full-run`; the launcher defaults
to `--sample`. Dry runs read metadata and rank parameters but do not load CNNs,
run image solvers/scoring, verify PNG decoding, or write output.

## Full processing, only when requested later

The implementation does not automatically submit jobs. For a future full run,
use `RUN_SCOPE=full` in an existing allocation or submit the launcher with
that export. Narrow each independent job by subject, ROI, generation model,
and optionally voxel rank range. Example (do not run for sample validation):

```bash
sbatch --export=ALL,RUN_SCOPE=full script/b2/postprocess_ranked_images.job \
    --subjects 2 --rois V1 --gen-models ADV_RN50 --resume
```

The time and memory requests are configurable resource requests, not measured
full-run runtime estimates. No full experiment was submitted during development.

## Output and restart behavior

The production root is `data/output/BrainDiVE/ICLR2027_top30`. Each group uses:

```text
sub-02/gen-ADV_RN50/split-01/roi-V1/voxel-rank-01__id-007733/
  unadjusted/
    top30_generation/rank-001__seed-N.png
    top30_ranking/rank-001__seed-N.png
    scores.csv
    diagnostics.jsonl
    failures.jsonl
    summary.json
  full_only/
    top30_generation/rank-001__seed-N.png
    top30_ranking/rank-001__seed-N.png
    scores.csv
    diagnostics.jsonl
    failures.jsonl
    summary.json
  full_and_prf/
    ...
```

`--top-n N` changes the count and folder names. Scores CSV includes all valid
candidates, both scores/rank positions, source paths, seed IDs, PNG statistics,
and pixel hashes. Diagnostics preserve solver and pre/post-clipping statistics.
Failures contain source paths, seeds, reasons, and solver details where available.
A failed image is skipped only for that branch. Missing/corrupt PNGs
and nonfinite scores are also logged. Model/configuration or infrastructure
errors stop the job instead of silently treating the entire dataset as invalid.
An output with zero valid images still has complete accounting and empty top-N
folders. Fewer than N valid images produce a reported shortfall, with no padding.

Only the union of both top-N pixel sets remains in memory. All scores are saved.
Each mode is published atomically after processing its voxel. Interrupted modes
restart from their original PNGs; completed modes can be verified with `--resume`.
Per-mode locks prevent concurrent writers. Resume checks include source metadata
and implementation hashes, ranking parameter checksum, source/checkpoint file
identities (path, size, modification time), solver options, and output checksums.
Large source images/backbones/generation checkpoints are identified by file stats,
not rehashed. Changed configurations require another output root; no overwrite
option is provided. Hidden leftover staging directories are not completed runs.

```bash
../../software/miniforge3/envs/env2025_v100/bin/python -B -m unittest tests.test_postprocess_ranked_images
```
