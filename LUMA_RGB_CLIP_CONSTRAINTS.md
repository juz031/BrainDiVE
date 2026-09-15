# Luma-only projection and RGB clipping

The concatenated pRF generation runner now defaults to
`--luma_contrast_method luma_rgb_clip`. Both Slurm launchers default to
`LUMA_CONTRAST_METHOD=luma_rgb_clip`. Select `joint_solver` for the previous
RGB adjustment and strict post-clipping matching. Existing callers of the shared
pipeline that do not select a method retain the legacy pipeline default; the
concat runner always passes its explicit selection.

## Timing and scope

Keep using the four contrast/luminance full-image/pRF timing arguments. For the
new method, contrast and luminance timing must match within each scope, and
enabled contrast must use `match`. pRF timing cannot start earlier than full
image timing. Existing independent timing and max/range behavior remains
available under `joint_solver` subject to its previous validation rules.

| Full-image mean/RMS | pRF mean/RMS | During guidance | After final decode |
|---|---|---|---|
| none | none | Off | Off |
| final | none | Off | Whole-image luma loop |
| final | final | Off | Coupled full/pRF luma solver |
| every_step | none | Whole-image luma loop | Whole-image luma loop |
| every_step | final | Whole-image luma loop | Coupled full/pRF luma solver |
| every_step | every_step | Coupled full/pRF luma solver | Coupled full/pRF luma solver |

No fallback to RGB adjustment occurs in the new method's pRF path.

## Algorithm and acceptance

1. Compute gamma-coded Rec.709 luma Y and store original color residual D=RGB-Y.
2. Full-only: follow the read-only reference's repeated luma normalization and
   scalar clipping, using the preceding iteration's clipped luma.
3. Dual scope: solve full-image/pRF mean and RMS together on grayscale luma,
   including clipping of trial luma. The numerical machinery from the existing
   joint solver is reused on three identical grayscale channels, not original RGB.
   This dual-scope algorithm is an extension beyond the whole-image reference.
4. Validate all active luma targets, restore the original D once (alpha=1), then
   clip RGB once. No post-RGB-clipping compensation is applied.

Existing tolerance flags use 0–255 units. Under the new method they are enforced
on the adjusted clipped luma **before color restoration and final RGB clipping**.
Post-RGB-clipping errors are recorded, not rejected. NaN/Inf pixels, invalid
shapes/ranges, and nonfinite generation/ranking scores remain rejection reasons.
Pre-RGB luma convergence failures are classified as
`luma_rgb_clip_solver_failure`, counted, and retried with the next available seed.
Thus accepted new-method PNGs do not guarantee final mean/RMS target matching.

## Controls

- `--luma_rgb_clip_match_iterations 20`: full-only normalization loop, maximum
  N+1 passes (21 by default), matching the reference. Stops early when both
  separate mean and RMS tolerances pass.
- `--dual_scope_every_step_iterations`: differentiable coupled pRF luma iterations.
- `--dual_scope_solver_max_nfev`: final coupled pRF luma evaluation limit.
- `--dual_scope_parameter_limit`, `--dual_scope_envelope_power`,
  `--luminance_shift_limit`: reused by the coupled pRF luma solver.

The reference's optional no-tolerance mode is intentionally not exposed: this
integration uses the existing separate contrast/luminance tolerances from the plan.

Slurm exports:

```bash
LUMA_CONTRAST_METHOD=luma_rgb_clip
LUMA_RGB_CLIP_MATCH_ITERATIONS=20
```

For example, keep full-image constraints during guidance and add pRF at final:

```bash
sbatch --export=ALL,LUMA_CONTRAST_METHOD=luma_rgb_clip,CONTRAST_FULL_IMAGE=every_step,LUMINANCE_FULL_IMAGE=every_step,CONTRAST_PRF=final,LUMINANCE_PRF=final \
  script/run_img_gen_prf_concat_model_rank.job
```

For old behavior, export `LUMA_CONTRAST_METHOD=joint_solver` instead.
Running Python directly requires specifying valid paired timing arguments when
selecting the new method; the old individual timing-argument defaults are retained.

## Saved output

Paths gain a method component between model pair and settings:

```text
SAVE_ROOT/sub-01/gen-...__rank-.../
  method-luma_rgb_clip/<settings>/seed-.../
    run_config.json
    generation_timing.json
    roi-hV4/voxel-.../ranking.json
```

`method-joint_solver` separates newly launched legacy-method runs as well.
Older unlabelled outputs are not modified. Tolerances and solver controls are
still not part of the path; use separate save roots for comparisons of those.

Run configuration and ranking schema 11 record `constraint_projection`,
including acceptance domain and effective solver per stage. Each saved image's
`luma_rgb_clip_diagnostics` records final pre/post-RGB-clipping statistics and
errors, alpha, clipping fraction/magnitude, and scalar solver details. The old
generic solver fields are retained for compatibility and identify the new method
explicitly. `guidance_constraint_diagnostics` records the number of successful
guidance projections, maximum pre/post errors, and the last projection details.
PNG statistics remain separate from float-image diagnostics.

The saved `unconstrained_images` are pre-final-adjustment decodes; when guidance
constraints were enabled, they are not fully unconstrained generation baselines.

## Verification

```bash
python -m unittest discover -s tests -t . -v
```

These CPU tests do not submit Slurm jobs or load diffusion weights. The numerical
reference-equivalence test reads the external reference source without modifying it.
