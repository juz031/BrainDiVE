# Optional image ranking

`--enable_ranking true|false` defaults to `true`. Both Slurm launchers expose
`ENABLE_RANKING`, also defaulting to `true`.

```bash
sbatch --export=ALL,ENABLE_RANKING=false script/run_img_gen_prf_concat_model_rank.job
```

With ranking enabled, the runner builds the ranking encoder, fits and saves the
per-voxel online ridge model, scores candidates with the generation and ranking
models, and selects the top `--ranking_top_n` by ranking-model score. The Python
default is 5; both launchers explicitly default to `all`.

With ranking disabled, ranking encoder/checkpoint metadata, training features,
NSD response training data and post-generation scoring are skipped. All
pixel/constraint-valid candidates are saved in seed-attempt order, without
sorting or score-based rejection. `--ranking_top_n` is ignored (logged).
Generation still uses the generation encoder/readout for brain guidance, and
constraint/pixel failures still reject seeds. A finite attempt budget can leave
fewer valid candidates than requested.

The selected `--image_save_mode` applies independently in either ranking mode.
It defaults to `both`. Unconstrained images mean pre-final-adjustment decodes,
not a separate unconstrained generation. Both `every_step` and `final` apply an
adjustment after the final decode, and both now capture the image before it.
`unconstrained_only` requires at least one enabled constraint; when all timings
are `none`, `both` saves only the pipeline output. Files are 8-bit RGB PNGs, so
they do not preserve the decoded float pixels exactly. `unconstrained_only`
skips the last adjustment entirely, along with its target-matching acceptance
checks. Guidance-time constraints remain active. Basic pixel checks and any
enabled score-finiteness checks remain active; ranking scores the unadjusted
decode in this mode. `both` and `constrained_only` retain the final adjustment
and score the adjusted output. There is no final-solver rejection or runtime
cost in `unconstrained_only` mode; guidance-time failures can still reject seeds.

New `run_config.json` files and per-voxel `ranking.json`/`generation.json` include
an `image_saving` block recording this provenance explicitly: capture stage,
constraint timings, whether guidance already used constraints, PNG quantization,
final-adjustment/rejection behavior, and the image version used for scoring.

New outputs are separated as follows (existing outputs are not moved):

```text
SAVE_ROOT/sub-01/
├── gen-DINO_RN50-concat__rank-OPEN_CLIP_RN50-concat/
│   └── ranking-enabled/method-.../<settings>/seed-.../
│       └── roi-V1/
│           ├── run_config.json
│           ├── generation_timing.json
│           └── voxel-rank-...__id-.../
│               ├── ranking.json
│               ├── ranking_model/
│               ├── seeds.json
│               ├── images/                 # when selected
│               └── unconstrained_images/   # when selected and available
└── gen-DINO_RN50-concat/
    └── ranking-disabled/method-.../<settings>/seed-.../
        └── roi-V1/
            ├── run_config.json
            ├── generation_timing.json
            └── voxel-rank-...__id-.../
                ├── generation.json
                ├── seeds.json
                ├── images/                 # when selected
                └── unconstrained_images/   # when selected and available
```

`generation.json` records the generation backbone, voxel/pRF, constraint settings,
image paths and diagnostics, attempts/valid counts/rejections, and timing summary.
It has no image scores, image rankings, ranking-model artifact references, or
top-N selection fields. Voxel selection ranks and pre-existing voxel R² remain:
these describe the chosen brain voxel, not a ranking of generated images.
`seeds.json` explicitly records whether validity included score checks.
GPU information remains in ROI timing files, not seed records.

The timing metric still excludes ranking/model-loading/file-saving overhead.
Switching off ranking can reduce overall runtime without directly changing what
the recorded diffusion-call timing measures. No existing jobs are changed.
