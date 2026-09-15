# Mean luma and contrast controls

All CLI targets/tolerances use 0-255 units. Mean luma is
`0.2126 R + 0.7152 G + 0.0722 B` on gamma-coded RGB, not physical luminance.

```bash
# Add these arguments to your usual concatenated-generation command:
--contrast_constraint_method match \
--contrast_constraint_full_image every_step \
--contrast_constraint_prf every_step \
--target_rms_contrast 57.48 \
--luminance_constraint_full_image final \
--luminance_constraint_prf final \
--ranking_top_n all
```

Omitted luminance targets for enabled scopes both use
`luminance_mean.byte_0_255.mean` from the subject's NSD statistics JSON.
This is the arithmetic mean of per-image whole-image luma, not its median
and not a separately measured pRF mean. The JSON defaults to
`<model_root>/S<subject_id>/nsd_contrast_statistics.json`; override it with
`--contrast_stats_json`. Generate it using `calculate_nsd_contrast_statistics.py
--subject N`, ensuring its output path matches the generation settings (both
scripts default to `/user_data/junruz/prf_models/concat/split_1_zscore`).
Explicit `--target_luminance_full_image` and `--target_luminance_prf` still
override their respective defaults. Metadata records the resolved targets and
their sources. Missing/invalid statistics are configuration errors when needed;
no statistics file is required for luminance when disabled or fully overridden.

Luminance defaults to `none` for both scopes.
Each constraint accepts `none`, `every_step`, or `final`;
`every_step` includes final projection. Within each statistic, pRF timing
requires whole-image timing at the same or earlier stage. Contrast and mean
luma schedules can differ. Combining enabled contrast with luminance requires
`match`; `max`/`range` remain available when luminance is disabled.

At each stage, a single projection fits all statistics active there. With all
four active, it uses the reference's local/global log scales and local/global
brightness offsets. Trials always start from the fixed input image, clip to
[0,1], and measure errors after clipping. The final fit uses SciPy float64;
every-step fitting uses differentiable PyTorch with backtracking. The latter
adds computation and autograd memory to each guidance evaluation.

`--contrast_validation_tolerance` and `--luminance_validation_tolerance` both
default to 0.01 byte, measured on unquantized floats. PNG statistics are
diagnostic; no saved-pixel target compensation is applied. Numerical failures
and unmet targets discard the seed and trigger replacement within the existing
attempt limit. Configuration and dependency errors remain fatal.

`--ranking_top_n` accepts a positive integer or `all`. All means every accepted,
scored candidate collected for that voxel, including a partial collection if
the attempt limit is reached. Images are named `seed-0123456789.png` (paired
pre-projection images use the same basename). `ranking.json` schema 10 records
`gen_rank` and `rank_model_rank`, independently computed over all fully valid
candidates for that voxel, descending score with 1 best. Ties use accepted
candidate order. Top-N selection and the order of `results` still follow the
ranking model; the legacy `rank` field aliases `rank_model_rank`.
`results` describes saved images; `candidate_rankings` records both scores/ranks
and a `saved` flag for every valid candidate, including unsaved candidates.
Existing output images are not automatically renamed or deleted. Use a new
save root to avoid mixing old and new naming conventions when reusing a seed plan.

Experiment paths include `__luma-full-<stage>-<target>` and
`__luma-prf-<stage>-<target>` (no target suffix when disabled), and `__keep-all`
or `__keep-<N>`. `images/` holds the selected projected images. When any
constraint is final, `unconstrained_images/` holds matching pre-final-projection
decodes. These may have come from a trajectory with constrained guidance;
metadata explicitly describes this and does not promise their RMS already
matches the target. `ranking_model/` continues holding `model.json` and
`parameters.npz`.

The Slurm launcher exposes `LUMINANCE_FULL_IMAGE`, `LUMINANCE_PRF`,
`TARGET_LUMINANCE_FULL_IMAGE`, `TARGET_LUMINANCE_PRF`,
`LUMINANCE_TOLERANCE_BYTE`, and `RANKING_TOP_N` via environment variables.
Luminance defaults to disabled and `RANKING_TOP_N` defaults to 30.

CPU regression checks (activate the project's environment first):

```bash
python -m unittest -v tests.test_joint_luma_integration tests.test_dual_scope_contrast_utils
```
