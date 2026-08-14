# pRF MEI Generation

Gradient-based maximally exciting image (MEI) generation for fitted pRF
encoding models. The implementation supports CLIP, OpenCLIP, DINO, and
adversarially trained ResNet backbones.

## Features

- Voxel-response maximization from fitted pRF checkpoints.
- Rec.709 luminance-based RMS contrast constraints.
- Independent contrast control during optimization, after optimizer steps,
  and before image export.
- Optional total-variation smoothing and configurable image jitter.
- Fixed-pRF cross-backbone readout fitting and validation.
- Contrast-matched natural-image validation with fresh feature extraction.

## Repository layout

```text
max_activation_utils.py               Core models and image optimization
maximize_prf_activation.py            MEI generation command-line interface
online_prf_fitting.py                  Fixed-pRF cross-backbone readout fitting
configs/                               Example experiment configurations
scripts/validate_fixed_prf_candidates.py
tests/                                 Unit tests
```

The repository contains source code and tests only. Model checkpoints,
stimulus images, generated images, fitted readouts, and experiment tables are
stored outside the repository.

## Requirements

- Python 3.11
- PyTorch
- NumPy
- Pillow
- PyTables
- The original `MEI` repository containing `prf_utils.py`
- Model checkpoints and backbone weights referenced by the selected config

The configured cluster environment is:

```bash
/home/hanfeig/venvs/mei-cu128/bin/python
```

## Quick smoke test

The smoke config performs two optimization steps for one V1 voxel and writes
all generated artifacts to `/user_data/hanfeig`:

```bash
bash /home/hanfeig/prf_mei_generation/scripts/run_smoke_test.sh
```

Expected output directory:

```text
/user_data/hanfeig/prf_max_activation/smoke_tests/prf_mei_generation/clip_rn50_v1_rms16
```

## MEI generation

Run either task in the longer comparison config:

```bash
/home/hanfeig/venvs/mei-cu128/bin/python maximize_prf_activation.py \
  --task_config configs/clip_rn50_v1_contrast.example.json \
  --task_id 0
```

Task `0` is unconstrained. Task `1` applies the reference contrast
configuration with target luminance mean 111 and RMS contrast 16.

The same settings can be supplied directly:

```bash
/home/hanfeig/venvs/mei-cu128/bin/python maximize_prf_activation.py \
  --source_repo /home/hanfeig/MEI \
  --model_name CLIP_RN50 \
  --layer_name layer2 \
  --model_dir /user_data/hanfeig/prf_models/fixed/split_1_zscore/S1/CLIP_RN50_set1/layer2 \
  --voxel_indices 9656 \
  --save_dir /user_data/hanfeig/prf_max_activation/manual_runs/clip_rn50_v1_voxel_9656 \
  --device cuda \
  --steps 100 \
  --jitter 0 \
  --contrast-units byte \
  --train-contrast 16 \
  --train-contrast-mode max \
  --post-step-contrast 16 \
  --post-step-contrast-mode max \
  --final-contrast 16 \
  --final-contrast-mode match \
  --final-contrast-mean 111
```

Contrast is the population standard deviation of Rec.709 luminance:

```text
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

`max` only reduces images above the requested contrast. `match` rescales each
image to the requested contrast and optionally to a requested luminance mean.

## Contrast-matched natural-image validation

The validation script compares retained MEIs against both original held-out
natural images and independently contrast-matched versions of those images.
Matched images are quantized to 8-bit and passed through the validator
backbone again; precomputed features from the original images are not reused.

```bash
/home/hanfeig/venvs/mei-cu128/bin/python \
  scripts/validate_fixed_prf_candidates.py \
  --ranking-root /user_data/hanfeig/prf_max_activation/ranking_candidates/S1/EXPERIMENT_NAME \
  --validator-model DINO_RN50 \
  --validator-split 2 \
  --batch-size 64
```

Validation output is written below the supplied ranking root.

## Tests

```bash
cd /home/hanfeig/prf_mei_generation
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=. /home/hanfeig/venvs/mei-cu128/bin/python \
  -m unittest discover -s tests -v
```
