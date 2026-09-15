# Generation regression tests

These eight test modules cover constraints, seed handling, ranking and image
saving, generation timing, NSD statistics, and SLURM launcher arguments.

From the BrainDiVE project root, with the project's Python environment active:

```bash
python -m unittest discover -s tests -t . -v
```

Run one module with its package-qualified name:

```bash
python -m unittest -v tests.test_seed_utils
```

The tests run on CPU with synthetic inputs and temporary files. They do not
submit SLURM jobs, load diffusion weights, or require the NSD dataset. The
reference-equivalence check in `test_luma_rgb_clip_integration.py` reads Hanfei's
external reference implementation when available and skips that check otherwise.

Launcher tests include explicit expectations for experiment defaults. If those
settings change deliberately, review the corresponding expectations rather than
changing experiment settings just to satisfy a test. At the time of this folder
move, the suite already had six launcher assertion failures across three test
methods (including four seed-mode subtests); these are not caused by relocation.

Report-specific tests remain in `fig_making/`; the separate analysis and pRF MEI
packages retain their own test folders.
