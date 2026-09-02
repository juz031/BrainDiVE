# BrainDiVE MEI image-statistics analysis

This package analyzes current MEIs, three RMS-controlled gradient conditions,
and two NSD validation-image selections using both full-image and
generation-pRF-weighted statistics. It intentionally excludes `pixel_raw`,
`maco_raw`, step-50 gradient snapshots, and FFT analyses.

## Current configuration

- subject: S1
- backbones: ADV-RN50 and DINO-RN50
- ROIs: V1, V2, V3, hV4
- voxels: ranks 1–20 per backbone/ROI
- conditions: current MEIs, `pixel_rms`, `maco_rms`,
  `maco_during_match`, NSD measured top 30, and NSD predicted top 30
- observations: 160 contexts × 6 conditions × 30 images = 28,800

The independent NSD prediction model is OpenCLIP ConvNeXt-Base concat set 2.
It uses its own set-2 pRF for prediction. All pRF-weighted image statistics and
display masks use the corresponding generation model's set-1 pRF.

## Local preparation and pilot

```bash
PY=/home/junruz/.local/envs/env2025/bin/python
CFG=/home/junruz/BrainDiVE/MEI_analysis/config/analysis_s1.yaml

$PY run_pipeline.py --config "$CFG" prepare
$PY run_pipeline.py --config "$CFG" validate
$PY run_pipeline.py --config "$CFG" pilot-task-ids
```

Run one feature work unit with:

```bash
$PY run_pipeline.py --config "$CFG" extract --feature basic --task-id 0 --device cpu
```

## Slurm

Preview the complete dependency graph without submitting:

```bash
bash slurm/submit_pipeline.sh --config config/analysis_s1.yaml --dry-run
```

Submit or resume:

```bash
bash slurm/submit_pipeline.sh --config config/analysis_s1.yaml --max-concurrent 40
```

Completed feature tasks are validated and skipped automatically. Pass
`--force` to recompute them. Every array index represents one
subject–backbone–ROI–voxel–condition work unit, normally containing 30 images.

To extend the analysis, add subjects, backbones, ROIs, conditions, or a larger
voxel rank limit in the YAML configuration. Backbones specify their current
image directory, gradient directory, and generation-pRF checkpoint directory;
conditions specify their image family and discovery/selection rule.
Input roots may use `{subject}`, `{subject_number}`, `{subject_number_02}`, or
`{bids_subject}` placeholders. For a multi-subject run, give the configuration
a shared, new `output_root`; the generated manifest and Slurm array expand to
all configured subjects automatically. Add another gradient method by adding a
`family: gradient` condition with its directory-level `method` name. The array
size is always read from the manifest, so no Slurm source edit is required.

## Outputs

Outputs are written under the configured `paths.output_root`, including input
manifests, pRF masks, restartable feature chunks, full and pRF-weighted feature
tables, voxel summaries, statistics, figures, logs, and synchronized HTML/PDF
reports.

After the final report, run the strict integrity check with:

```bash
$PY run_pipeline.py --config "$CFG" validate --require-analysis
```

The Slurm dependency graph runs this strict form automatically as its final
stage.
