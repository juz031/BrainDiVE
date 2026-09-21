# Standalone ranking-readout training

`train_ranking_model_concat_prf.py` trains ranking readouts only. It does not
load a CNN/diffusion model, extract features, generate images or search pRFs.
The original generation runner is unchanged and **does not yet load these saved
readouts**; enabling ranking there still fits its readout online.

## Selection and fit

Select voxels by generation-model R² within each ROI, using the evaluation's
noise-ceiling-filtered ordering. Translate to global big-mask voxel indices and
use **each selected voxel's generation-model pRF**. Ranking features at this pRF
are concatenated in ranking metadata layer order. The ranking model's own best
pRFs/weights are never used. Retino labels override duplicate Kastner names;
V1/V2/V3 combine dorsal and ventral labels, FFA combines FFA-1/FFA-2.

The fit matches the existing runner: train+nested normalization, train-only
weights, 20 lambdas selected by nested SSE, regularized intercept, epsilon 1e-4,
and the existing float32 Cholesky/direct solver. Validation responses are not
used. One voxel is fitted at a time. The lightweight helper mirrors the original
definitions; regression tests guard against drift without importing diffusion.

## CLI defaults

| Argument | Default |
| --- | --- |
| `--subject_id` | Required; 1–8 |
| `--gen_model` | `DINO_RN50` |
| `--rank_model` | `OPEN_CLIP_RN50` |
| `--split_id` | `1`; choices 1/2 |
| `--roi` | `hV4`; accepts multiple names |
| `--k` | `20` per ROI |
| `--voxel_rank_range START END` | Unset; inclusive, overrides k |
| `--r2_noise_ceiling_threshold` | `0.0`; must match evaluation |
| `--nsd_root` | `/ocean/projects/soc250009p/shared/datasets/nsd_preproc` |
| `--roi_path` | `NSD_ROOT/rois/S<ID>_voxel_roi_info.npy` |
| `--model_root` | `/ocean/projects/soc250009p/jzhao7/data/model/concat/split_1_zscore` |
| `--feature_root` | `/ocean/projects/soc250009p/jzhao7/data/prf_features` |
| `--save_root` | `/ocean/projects/soc250009p/jzhao7/data/model/ranking` |
| `--device` | `auto`; also cuda/cpu |
| `--skip_existing` | Off; verify complete matching artifacts before skipping |
| `--overwrite` | Off; explicitly replace selected artifacts |
| `--dry_run` | Off; no training, CUDA initialization or output writes |

Dry runs check split indices, ROI/R²/NC ordering, pRF membership, image-row
alignment, feature headers and output policy. They do not scan all feature
values or run a GPU computation. Missing-response subjects are rejected rather
than silently dropping rows and misaligning features/splits. Only trusted local
checkpoint/NSD files should be loaded: the existing formats use pickle.

## B2 launcher

`script/b2/train_ranking_model_concat_prf.job` defaults to subject 2, split 1,
hV4 top 20, one L40S-48 GPU, 5 CPUs, 32 GB RAM and a 10-hour walltime request.
This is not a measured runtime estimate. The two array tasks are
DINO_RN50 → OPEN_CLIP_RN50 and ADV_RN50 → OPEN_CLIP_RN50.

From the BrainDiVE repository root:

```bash
# Read-only preflight (task 0 by default), no scheduler submission.
bash script/b2/train_ranking_model_concat_prf.job 2 --dry_run

# Each submission runs the two model-pair tasks for one subject.
sbatch script/b2/train_ranking_model_concat_prf.job 2
sbatch script/b2/train_ranking_model_concat_prf.job 5
sbatch script/b2/train_ranking_model_concat_prf.job 7

sbatch --export=ALL,SPLIT_ID=2,ROI=V1,VOXEL_RANK_START=1,VOXEL_RANK_END=10,SKIP_EXISTING=true \
    script/b2/train_ranking_model_concat_prf.job 7
```

The positional subject overrides `SUBJECT_ID`. Other environment overrides:
`GEN_MODEL`, `RANK_MODEL`, `SPLIT_ID`, `ROI`, `K`, `VOXEL_RANK_START`,
`VOXEL_RANK_END`, `R2_NOISE_CEILING_THRESHOLD`, `NSD_ROOT`, `ROI_PATH`,
`MODEL_ROOT`, `FEATURE_ROOT`, `SAVE_ROOT`, `CONDA_ENV`, `DEVICE`,
`SKIP_EXISTING`, `OVERWRITE`, `DRY_RUN`.
The flags accept `true`/`false`; rank endpoints must be supplied together.
Use `--array=0` when overriding a model pair, to avoid duplicate tasks.
GPU selection belongs in `#SBATCH --gres` or an `sbatch --gres` override;
V100 also requires `CONDA_ENV=env2025_v100`.
Extra Python arguments may follow the subject. Outside an allocation the shell
launcher permits dry runs only. Ensure `script/b2/slurm_out` exists before sbatch.

## Save layout and restart safety

```text
/ocean/projects/soc250009p/jzhao7/data/model/ranking/
├── .locks/                 # persistent advisory locks, not models
└── S2/
    └── gen-DINO_RN50__rank-OPEN_CLIP_RN50/
        └── split-01/
            └── roi-hV4/
                └── voxel-rank-01__id-NNNNNN/
                    ├── parameters.npz
                    └── model.json
```

Voxel IDs are zero-based big-mask indices, padded to six digits; ranks are
one-based. Only requested subject/pair/split/ROI paths are created. NPZ contains
weights, intercept, feature mean/std, channels, lambda selection, nested SSE and
exact split indices. JSON preserves the existing artifact schema and adds
standalone provenance/configuration, input identities, runtime and checksums.
pRF IDs and coordinates are stored in JSON, not a separate directory level.

Each voxel is staged separately; JSON is published last with the NPZ checksum.
An interrupted two-file replacement is detected as incomplete/corrupt, not
silently skipped. Per-output advisory locks reject simultaneous writers. A
killed process may leave hidden staging folders; these are never treated as
models. Neither original encoding checkpoints nor extracted features are changed.

Skip and overwrite are mutually exclusive. Without either, existing selected
outputs cause an error before training starts. Partial/mismatched outputs require
explicit overwrite or a different save root. Small source files and implementation
files are hashed; large feature/beta files are identified by absolute path,
size and nanosecond modification time rather than full-content hashes. Moving or
touching inputs conservatively invalidates skipping. Device/hardware is recorded
but is not a compatibility key, so completed models may be resumed on another GPU.

```bash
python -B -m unittest tests.test_standalone_ranking_training
```
