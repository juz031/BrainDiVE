"""Exercise launcher argument wiring without Slurm, CUDA, or generation."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [ROOT / 'script' / f'{prefix}_img_gen_prf_concat_model_rank.job'
           for prefix in ('run', 'test')]


def runner_parser():
    """Read current CLI flag names, arities and choices without importing models."""
    tree = ast.parse((ROOT / 'run_image_generation_concat_pRF_model_ranking.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    parser = argparse.ArgumentParser()
    for stmt in main.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'add_argument':
            continue
        options = {}
        for kw in call.keywords:
            if kw.arg in ('choices', 'nargs', 'action', 'dest'):
                options[kw.arg] = ast.literal_eval(kw.value)
            elif kw.arg == 'type' and isinstance(kw.value, ast.Name):
                options['type'] = {'int': int, 'float': float}.get(kw.value.id, str)
        parser.add_argument(*(ast.literal_eval(a) for a in call.args), **options)
    return parser


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = runner_parser()

    def launch(self, script, *, expect_ok=True, **overrides):
        env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ}
        env.update(SLURM_JOBID='test', SLURM_ARRAY_TASK_ID='0', SLURM_JOB_NODELIST='mock')
        env.update({k: str(v) for k, v in overrides.items()})
        # Intercept environment activation, GPU inspection and Python invocation.
        # The launchers' own Bash configuration/validation still executes normally.
        harness = r'''
source() { :; }
nvidia-smi() { :; }
python() { printf '\0ARGV\0'; printf '%s\0' "$@"; }
. "$1"
'''
        result = subprocess.run(['bash', '-c', harness, 'launcher-test', str(script)],
                                env=env, capture_output=True)
        self.last_launch_log = result.stdout.split(b'\0ARGV\0', 1)[0].decode()
        if not expect_ok:
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(b'\0ARGV\0', result.stdout)
            return
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        args = result.stdout.split(b'\0ARGV\0', 1)[1].decode().strip('\0').split('\0')
        self.assertEqual(args[:2], ['-u', 'run_image_generation_concat_pRF_model_ranking.py'])
        return self.parser.parse_args(args[2:])

    def test_bash_syntax(self):
        for script in SCRIPTS:
            subprocess.run(['bash', '-n', str(script)], check=True)

    def test_ranking_toggle(self):
        for script in SCRIPTS:
            self.assertEqual(self.launch(script).enable_ranking, 'false' if script.name.startswith('test') else 'true')
            for mode in ('true', 'false'):
                args = self.launch(script, ENABLE_RANKING=mode, RANKING_TOP_N=3)
                self.assertEqual(args.enable_ranking, mode)
                if mode == 'false':
                    self.assertIn('ranking_top_n ignored', self.last_launch_log)
            self.launch(script, ENABLE_RANKING='invalid', expect_ok=False)

    def test_unconstrained_image_save_switch(self):
        for script in SCRIPTS:
            self.assertEqual(self.launch(script).image_save_mode, 'unconstrained_only')
            for value in ('unconstrained_only', 'constrained_only', 'both'):
                args = self.launch(script, IMAGE_SAVE_MODE=value,
                                   CONTRAST_PRF='final', LUMINANCE_PRF='final')
                self.assertEqual(args.image_save_mode, value)
                if value == 'unconstrained_only':
                    self.assertIn('adjustment skipped (unconstrained_only)', self.last_launch_log)
            self.launch(script, IMAGE_SAVE_MODE='invalid', expect_ok=False)
            self.launch(script, IMAGE_SAVE_MODE='unconstrained_only',
                        CONTRAST_PRF='every_step', LUMINANCE_PRF='every_step')
            self.launch(script, IMAGE_SAVE_MODE='unconstrained_only',
                        CONTRAST_FULL_IMAGE='none', CONTRAST_PRF='none',
                        LUMINANCE_FULL_IMAGE='none', LUMINANCE_PRF='none', expect_ok=False)

    def test_unconstrained_save_requires_enabled_constraint_and_opt_in(self):
        tree = ast.parse((ROOT / 'run_image_generation_concat_pRF_model_ranking.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'save_unconstrained_images'
                                  for t in n.targets))
        expression = compile(ast.Expression(assignment.value), '<save switch>', 'eval')
        for value in ('unconstrained_only', 'constrained_only', 'both'):
            for stage in ('none', 'every_step', 'final'):
                enabled = eval(expression, dict(args=argparse.Namespace(image_save_mode=value),
                                               constraint_stages=(stage, 'none')))
                self.assertEqual(enabled, value != 'constrained_only' and stage != 'none')
        helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'unconstrained_candidate_image')
        scope = dict(save_unconstrained_images=False)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), '<raw helper>', 'exec'), scope)
        # Disabled saving returns before image conversion or statistics computation.
        self.assertEqual(scope['unconstrained_candidate_image'](object()), (None, None))

    def test_preserved_experiment_defaults(self):
        for script in SCRIPTS:
            args = self.launch(script)
            small = script.name.startswith('test')
            self.assertEqual(args.num_seeds, 10 if small else 1000)
            self.assertEqual(args.ranking_top_n, 'all')
            self.assertEqual(args.voxel_rank_range, [1, 3] if small else [1, 20])
            self.assertEqual(args.seed_mode, 'file' if small else 'fixed')
            self.assertEqual(args.contrast_constraint_full_image, 'every_step')
            self.assertEqual(args.contrast_constraint_prf, 'final')
            self.assertEqual(args.luminance_constraint_full_image, 'every_step')
            self.assertEqual(args.luminance_constraint_prf, 'final')
            self.assertEqual(args.contrast_validation_tolerance, 1 if small else 3)
            self.assertEqual(args.luminance_validation_tolerance, 1 if small else 3)
            self.assertEqual(args.max_generation_attempts, None if small else 1200)
            self.assertEqual(args.luma_contrast_method, 'luma_rgb_clip')
            self.assertEqual(args.luma_rgb_clip_match_iterations, 30 if small else 50)

    def test_seed_modes_and_backbone_scales(self):
        for script in SCRIPTS:
            for task, scale in ((0, 300), (1, 300)):
                # The full launcher's user-selected generator-seed default 42
                # makes only fixed mode usable without changing that assignment.
                for mode in (('random', 'fixed', 'file') if script.name.startswith('test') else ('fixed',)):
                    with self.subTest(script=script.name, task=task, mode=mode):
                        env = dict(SEED_MODE=mode, SLURM_ARRAY_TASK_ID=task)
                        if mode == 'fixed':
                            env['SEED_GENERATOR_SEED'] = '42'
                        elif mode == 'file':
                            env.update(SEED_FILE=__file__, SEED_FILE_KEY='valid_seeds')
                        args = self.launch(script, **env)
                        self.assertEqual(args.brain_guidance_scale, scale)
                        self.assertEqual(args.seed_mode, mode)
                        self.assertEqual(args.seed_file, __file__ if mode == 'file' else None)
                        self.assertEqual(args.seed_generator_seed, 42 if mode == 'fixed' else None)

    def test_timing_tolerances_and_solver_overrides(self):
        for script in SCRIPTS:
            for stage in ('none', 'final', 'every_step'):
                args = self.launch(script, IMAGE_SAVE_MODE='both', SEED_MODE='fixed', SEED_GENERATOR_SEED=42, CONTRAST_FULL_IMAGE=stage,
                                   CONTRAST_PRF=stage, LUMINANCE_FULL_IMAGE=stage,
                                   LUMINANCE_PRF=stage, CONTRAST_TOLERANCE_BYTE='.1',
                                   LUMINANCE_TOLERANCE_BYTE='.2', MAX_GENERATION_ATTEMPTS=1200,
                                   DUAL_SCOPE_SOLVER_MAX_NFEV=600,
                                   DUAL_SCOPE_EVERY_STEP_ITERATIONS=20,
                                   DUAL_SCOPE_PARAMETER_LIMIT=7, DUAL_SCOPE_ENVELOPE_POWER=.6,
                                   LUMINANCE_SHIFT_LIMIT=.8)
                self.assertEqual(args.contrast_validation_tolerance, .1)
                self.assertEqual(args.luminance_validation_tolerance, .2)
                self.assertEqual(args.contrast_constraint_full_image, stage)
                self.assertEqual(args.luminance_constraint_prf, stage)
                self.assertEqual(args.max_generation_attempts, 1200)
                self.assertEqual(args.dual_scope_solver_max_nfev, 600)
                self.assertEqual(args.dual_scope_every_step_iterations, 20)
                self.assertEqual(args.dual_scope_parameter_limit, 7)
                self.assertEqual(args.dual_scope_envelope_power, .6)
                self.assertEqual(args.luminance_shift_limit, .8)

    def test_empty_luminance_targets_use_dataset(self):
        for script in SCRIPTS:
            args = self.launch(script, SEED_MODE='fixed', SEED_GENERATOR_SEED=42,
                               CONTRAST_FULL_IMAGE='final', CONTRAST_PRF='final',
                               LUMINANCE_FULL_IMAGE='final', LUMINANCE_PRF='final', TARGET_LUMINANCE_FULL_IMAGE='',
                               TARGET_LUMINANCE_PRF='')
            # The full launcher's explicit :-116 defaults remain unchanged.
            expected = None if script.name.startswith('test') else 116
            self.assertEqual(args.target_luminance_full_image, expected)
            self.assertEqual(args.target_luminance_prf, expected)

    def test_subject_paths_and_experiment_overrides(self):
        for script in SCRIPTS:
            args = self.launch(script, SEED_MODE='fixed', SEED_GENERATOR_SEED=42, SUBJECT_ID=2, SPLIT_ID=2,
                               MODEL_ROOT='/tmp/models with spaces', FEATURE_ROOT='/tmp/features',
                               SAVE_ROOT='/tmp/results with spaces', NUM_SEEDS=12, NUM_STEPS=8,
                               VOXEL_RANK_START=6, VOXEL_RANK_END=7, RANKING_TOP_N='all',
                               ROI='V1', SAG_SCALE=.5, BRAIN_GUIDANCE_SCALE=123)
            self.assertEqual(args.subject_id, 2)
            self.assertEqual(args.split_id, 2)
            self.assertEqual(args.contrast_stats_json,
                             '/tmp/models with spaces/S2/nsd_contrast_statistics.json')
            self.assertEqual(args.feature_root, '/tmp/features')
            self.assertEqual(args.save_root, '/tmp/results with spaces')
            self.assertEqual(args.num_seeds, 12)
            self.assertEqual(args.num_steps, 8)
            self.assertEqual(args.voxel_rank_range, [6, 7])
            self.assertEqual(args.roi, ['V1'])
            self.assertEqual(args.sag_scale, .5)
            self.assertEqual(args.brain_guidance_scale, 123)

    def test_invalid_settings_fail_before_generation(self):
        for script in SCRIPTS:
            for env in (
                dict(SEED_MODE='file', SEED_FILE=''),
                dict(SEED_MODE='random', SEED_FILE=__file__),
                dict(SEED_MODE='random', SEED_GENERATOR_SEED=42),
                dict(SEED_MODE='typo'), dict(SLURM_ARRAY_TASK_ID=-1),
                dict(SLURM_ARRAY_TASK_ID=2), dict(RANKING_TOP_N=0),
                dict(CONTRAST_FULL_IMAGE='none', CONTRAST_PRF='final'),
                dict(LUMINANCE_FULL_IMAGE='none', LUMINANCE_PRF='final'),
            ):
                with self.subTest(script=script.name, env=env):
                    self.launch(script, expect_ok=False, **env)

    def test_new_method_flags_and_legacy_option(self):
        for script in SCRIPTS:
            for method in ('luma_rgb_clip','joint_solver'):
                args=self.launch(script,LUMA_CONTRAST_METHOD=method,LUMA_RGB_CLIP_MATCH_ITERATIONS=40)
                self.assertEqual(args.luma_contrast_method,method)
                self.assertEqual(args.luma_rgb_clip_match_iterations,40)
            self.launch(script,expect_ok=False,LUMA_CONTRAST_METHOD='typo')
            self.launch(script,expect_ok=False,LUMINANCE_FULL_IMAGE='final',LUMINANCE_PRF='final')
            args=self.launch(script,LUMA_CONTRAST_METHOD='joint_solver',
                             LUMINANCE_FULL_IMAGE='final',LUMINANCE_PRF='final')
            self.assertEqual(args.luminance_constraint_full_image,'final')

    def test_new_method_logs_actual_stage_routing(self):
        for script in SCRIPTS:
            for full, prf, guidance, final in (
                ('none','none','no constraints','no constraints'),
                ('final','none','no constraints','full-image luma normalization; max passes=21'),
                ('every_step','none','full-image luma normalization; max passes=21',
                 'full-image luma normalization; max passes=21'),
                ('every_step','final','full-image luma normalization; max passes=21',
                 'coupled full-image+pRF luma solver; max evaluations=300'),
                ('final','final','no constraints',
                 'coupled full-image+pRF luma solver; max evaluations=300'),
                ('every_step','every_step','coupled full-image+pRF luma solver; max iterations=30',
                 'coupled full-image+pRF luma solver; max evaluations=300'),
            ):
                with self.subTest(script=script.name, full=full, prf=prf):
                    self.launch(script, CONTRAST_FULL_IMAGE=full,LUMINANCE_FULL_IMAGE=full,
                                IMAGE_SAVE_MODE='both',
                                CONTRAST_PRF=prf,LUMINANCE_PRF=prf,
                                LUMA_RGB_CLIP_MATCH_ITERATIONS=20,
                                DUAL_SCOPE_EVERY_STEP_ITERATIONS=30)
                    self.assertIn('During guidance: '+guidance,self.last_launch_log)
                    self.assertIn('After final decode: '+final,self.last_launch_log)
                    self.assertIn('post-RGB-clipping errors recorded only',self.last_launch_log)
                    self.assertIn('--luma_contrast_method luma_rgb_clip',self.last_launch_log)
                    self.assertIn('method-luma_rgb_clip',self.last_launch_log)
            for invalid in ('0','-1','abc','2.5'):
                self.launch(script,expect_ok=False,LUMA_RGB_CLIP_MATCH_ITERATIONS=invalid)


if __name__ == '__main__':
    unittest.main()
