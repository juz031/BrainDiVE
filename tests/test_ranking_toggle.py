"""Exercise actual runner branches without initializing diffusion or encoders."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from seed_utils import save_seed_record, summarize_generation_times
from tests.test_joint_luma_integration import source_tree, run_nodes


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


class RankingToggleTests(unittest.TestCase):
    def setUp(self):
        self.tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')

    def test_disabled_skips_all_ranking_resources_and_scoring(self):
        # All ranking-only resource calls must be protected by the toggle.
        parents = {child: node for node in ast.walk(self.tree) for child in ast.iter_child_nodes(node)}
        checked = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            target = name in ('rank_model_fitting', 'save_online_ranking_model',
                              'score_generated_images_for_voxel', 'load_nsd_data')
            if name in ('build_brain_encoder', 'load_concat_metadata'):
                target = 'args.rank_model' in ast.unparse(node)
            if not target:
                continue
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.If) and ast.unparse(parent.test) == "args.enable_ranking == 'true'":
                    break
                parent = parents.get(parent)
            self.assertIsNotNone(parent, f'Unguarded ranking resource: {name}')
            # No model/file functions are supplied: executing a disabled gate
            # must not resolve or call any of them.
            run_nodes([parent], dict(args=SimpleNamespace(enable_ranking='false')))
            checked.add(name)
        self.assertTrue({'build_brain_encoder', 'load_concat_metadata', 'rank_model_fitting',
                         'save_online_ranking_model', 'score_generated_images_for_voxel',
                         'load_nsd_data'} <= checked)
        accept = next(n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "args.enable_ranking == 'false'"
                      and 'image_records.extend' in ast.unparse(n))
        scope = dict(args=SimpleNamespace(enable_ranking='false'), image_records=[],
                     pending_records=[{'seed': 90}, {'seed': 12}])
        run_nodes([accept], scope)
        self.assertEqual(scope['image_records'], [{'seed': 90}, {'seed': 12}])

    def test_disabled_real_png_and_json_all_save_modes(self):
        loop = next(n for n in ast.walk(self.tree) if isinstance(n, ast.For)
                    and any(assigned(s, 'ranked_indices') for s in n.body))
        start = next(i for i, n in enumerate(loop.body) if assigned(n, 'ranked_indices'))
        end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.With)
                   and 'generation_summary_for_mode' in ast.unparse(n)) + 1
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'generation_summary_for_mode')
        for mode in ('unconstrained_only', 'constrained_only', 'both'):
            with tempfile.TemporaryDirectory() as directory:
                images = None if mode == 'unconstrained_only' else os.path.join(directory, 'images')
                raw = None if mode == 'constrained_only' else os.path.join(directory, 'unconstrained_images')
                for folder in (images, raw):
                    if folder:
                        os.mkdir(folder)
                args = SimpleNamespace(enable_ranking='false', ranking_top_n=1, num_seeds=3,
                    gen_model='DINO_RN50', rank_model='MISSING_MODEL', split_id=1, seed_mode='fixed',
                    contrast_validation_tolerance=1., contrast_constraint_method='match',
                    contrast_constraint_full_image='every_step', contrast_constraint_prf='final',
                    image_save_mode=mode, target_rms_contrast=57.48, dual_scope_solver_max_nfev=300,
                    dual_scope_every_step_iterations=30, dual_scope_parameter_limit=8., dual_scope_envelope_power=.5)
                records = [dict(seed=seed, tensor=torch.full((3, 2, 2), .5),
                    raw_tensor=torch.full((3, 2, 2), .4), rms_contrast_byte=1., raw_rms_contrast_byte=1.,
                    contrast_diagnostics={}, dual_scope_solver=None, joint_luma_rms_solver=None,
                    guidance_constraint_diagnostics={}) for seed in (90, 12)]
                scope = dict(args=args, image_records=records, images_folder=images,
                    image_saving_metadata={'mode': mode},
                    unconstrained_images_folder=raw, voxel_folder=directory, np=np, torch=torch,
                    os=os, json=json, Image=Image, gen_layer_names=['relu'], region='V1', ss=1,
                    voxel_id=8, idx=0, voxel_rank=1, topk_values=[.8], i=0, prf_idx=0,
                    prf_grid_name='test', prf_grid_params=np.array([[0., 0., .1]]),
                    contrast_prf_layer='relu', contrast_prf_native_size=112, attempted_seeds=[90, 7, 12],
                    seed_generation_times=[], summarize_generation_times=summarize_generation_times,
                    seed_plan={'plan_id': 'test'}, max_generation_attempts=3, acceptance_domain='test',
                    constraint_method_metadata={}, luminance_metadata={}, save_constrained_images=images is not None,
                    save_unconstrained_images=raw is not None, unconstrained_image_content='pre_final',
                    rejected_candidates={'luma_rgb_clip_solver_failure': 1, 'nonfinite_gen_score': 0,
                                         'nonfinite_rank_score': 0})
                run_nodes([helper, *loop.body[start:end]], scope)
                result = json.loads((Path(directory) / 'generation.json').read_text())
                self.assertFalse(result['ranking_enabled'])
                self.assertEqual(result['saved_image_count'], 2)  # ignores top_n=1
                self.assertEqual([r['seed'] for r in result['results']], [90, 12])
                self.assertEqual(result['candidate_shortfall'], 1)
                for key in ('ranking_definitions', 'candidate_rankings', 'ranking_top_n', 'online_ranking_model'):
                    self.assertNotIn(key, result)
                for row in result['results']:
                    for key in ('rank', 'gen_rank', 'rank_model_rank', 'gen_score', 'rank_score'):
                        self.assertNotIn(key, row)
                    for key, folder in (('image_path', images), ('raw_image_path', raw)):
                        self.assertEqual(row[key] is None, folder is None)
                        if folder:
                            self.assertTrue(Path(row[key]).is_file())
                self.assertFalse((Path(directory) / 'ranking.json').exists())
                self.assertFalse((Path(directory) / 'ranking_model').exists())

    def test_seed_validity_definition_without_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seeds.json'
            save_seed_record(path, {}, [90, 12], [90], ranking_enabled=False)
            data = json.loads(path.read_text())
            self.assertFalse(data['ranking_enabled'])
            self.assertIn('without_post_generation_scoring', data['valid_seed_definition'])

    def test_image_provenance_metadata(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'build_image_save_metadata')
        scope = {}
        run_nodes([helper], scope)
        for mode in ('both', 'unconstrained_only', 'constrained_only'):
            for stage in ('none', 'every_step', 'final'):
                for ranking in (True, False):
                    data = scope['build_image_save_metadata'](mode, (stage,) * 4, ranking)
                    self.assertEqual(data['pre_adjustment']['saving_enabled'],
                                     mode != 'constrained_only' and stage != 'none')
                    self.assertEqual(data['pre_adjustment']['guidance_already_used_constraints'],
                                     stage == 'every_step')
                    self.assertEqual(data['save_mode_skips_final_adjustment'], mode == 'unconstrained_only')
                    self.assertEqual(data['final_adjustment_enabled'],
                                     stage != 'none' and mode != 'unconstrained_only')
                    self.assertFalse(data['final_adjustment_rejected_seeds_saved'])
                    self.assertFalse(data['exact_float_decode_preserved'])
                    self.assertEqual(data['bits_per_channel'], 8)
                    self.assertEqual(data['scores_computed_on'] is None, not ranking)

    def test_ranking_output_directories_are_separate(self):
        names = ('model_folder', 'settings_folder', 'experiment_folder')
        nodes = [n for n in ast.walk(self.tree) if any(assigned(n, name) for name in names)]
        paths = []
        for enabled in ('true', 'false'):
            args = SimpleNamespace(enable_ranking=enabled, gen_model='DINO_RN50', rank_model='OPEN_CLIP_RN50',
                split_id=1, contrast_constraint_full_image='every_step', contrast_constraint_prf='final',
                num_seeds=10, ranking_top_n=3, luma_contrast_method='luma_rgb_clip',
                save_root='/tmp/test-ranking-paths', subject_id=1, seed_mode='fixed')
            scope = dict(args=args, path_component=str, os=os, num_steps=100, brain_guidance_scale=250,
                         sag_scale=.75, contrast_folder='match-rms57.48', luma_full='every_step-116',
                         luma_prf='final-116', seed_plan={'plan_id': 'test'})
            run_nodes(nodes, scope)
            path = scope['experiment_folder']
            paths.append(path)
            self.assertIn('ranking-enabled' if enabled == 'true' else 'ranking-disabled', path)
            self.assertIn('__keep-3' if enabled == 'true' else '__keep-all', path)
            self.assertEqual('OPEN_CLIP' in path, enabled == 'true')
        self.assertNotEqual(*paths)

    def test_run_config_is_roi_specific(self):
        node = next(n for n in ast.walk(self.tree) if assigned(n, 'config_file'))
        paths = []
        for region in ('V1', 'V2', 'hV4'):
            scope = dict(os=os, region_folder=f'/tmp/experiment/roi-{region}')
            run_nodes([node], scope)
            self.assertEqual(scope['config_file'], f'/tmp/experiment/roi-{region}/run_config.json')
            paths.append(scope['config_file'])
        self.assertEqual(len(set(paths)), 3)
        config = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == 'invocation_regions' for k in n.keys))
        fields = {k.value: v for k, v in zip(config.keys, config.values) if isinstance(k, ast.Constant)}
        scope = dict(region='V2', args=SimpleNamespace(roi=['V1', 'V2']))
        self.assertEqual(eval(compile(ast.Expression(fields['regions']), '<config>', 'eval'), scope), ['V2'])


if __name__ == '__main__':
    unittest.main()
