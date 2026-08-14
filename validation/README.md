# Independent validation

`heldout_model.py` evaluates saved BrainDiVE images without changing generation
or ranking.

The default evaluator is:

- backbone: OpenCLIP ConvNeXt-Base;
- backbone pretraining set: LAION-400M;
- voxel encoding fit: `set2`;
- layer mapping: RN50 `layer2` to ConvNeXt `stage2`, and RN50 `layer3` to
  ConvNeXt `stage3`.

This differs from the current generation model (OpenCLIP RN50, YFCC-15M,
`set1`) and all current RN50 ranking models in both backbone architecture and
pretraining set. The encoding model is also fit on the complementary NSD image
split.

Available validation models are:

- `OPEN_CLIP_CONVNEXT_BASE`
- `CLIP_RN50`
- `OPEN_CLIP_RN50`
- `DINO_RN50`
- `SIMCLR_RN50`
- `ADV_RN50`

Select a model and optionally a layer:

```bash
python -u validation/heldout_model.py \
  --validation-model DINO_RN50 \
  --validation-layer layer3 \
  --validation-set 2 \
  --device cuda
```

If `--validation-layer` is omitted, the script maps the generation layer to
the closest available resolution in the selected backbone. For example,
RN50 `layer2` maps to ConvNeXt `stage2`, while it remains `layer2` for another
RN50 model.

Independence is checked against both generation and ranking models using model
identity, backbone family, pretraining dataset, and encoding-fit set. A
selection that is not fully independent stops with the failed checks listed.
Use `--allow-nonindependent` only for deliberate comparison/control analyses.

Run all saved outputs:

```bash
cd /home/junruz/BrainDiVE
source activate env2025
python -u validation/heldout_model.py \
  --input /user_data/junruz/BrainDiVE/contrast_con/pRF_zscore_model_ranked/sub-01 \
  --validation-model OPEN_CLIP_CONVNEXT_BASE \
  --validation-set 2 \
  --device cuda \
  --overwrite
```

Or submit:

```bash
sbatch script/run_independent_validation.job
```

The Slurm launcher also accepts environment overrides:

```bash
VALIDATION_MODEL=DINO_RN50 \
VALIDATION_LAYER=layer3 \
VALIDATION_SET=2 \
sbatch script/run_independent_validation.job
```

Check metadata and required files without loading the network:

```bash
python validation/heldout_model.py --dry-run
```

For every voxel, validation is saved beside `ranking.json` under:

```text
validation/OPEN_CLIP_CONVNEXT_BASE-stageN-set2.json
```

The filename changes with the selected model, layer, and encoding-fit set, for
example `validation/DINO_RN50-layer3-set2.json`.

Each run also receives JSON and CSV summaries in its top-level `validation`
directory. Original `ranking.json` files and ranked images are not modified.

The validator scores the images retained in `ranking.json`. It does not
validate discarded candidates that were never saved.
