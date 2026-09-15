"""CPU numerical and runner regression tests without loading diffusion weights.

AST extraction executes the actual pipeline/runner blocks while avoiding
their heavyweight encoder imports. No copied projection or saving logic.
"""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import dual_scope_contrast_utils as dual
import joint_luma_contrast_utils as joint
import luma_rgb_clip_utils as luma_clip
from seed_utils import build_seed_plan
import seed_utils

ROOT = Path(__file__).resolve().parents[1]


def source_tree(filename):
    return ast.parse((ROOT / filename).read_text())


def run_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<project source>", "exec"), namespace)


def pipeline_namespace():
    namespace = {**vars(dual), **vars(joint), **vars(luma_clip), "torch": torch, "np": np}
    tree = source_tree("brain_guide_pipeline.py")
    names = {"_validate_contrast_constraint", "_validate_contrast_constraint_stages",
             "apply_rms_contrast_constraint", "prepare_brain_guidance_image"}
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in names
    ) or (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and "CONTRAST" in t.id for t in n.targets))]
    run_nodes(nodes, namespace)
    return namespace


def images_and_prf(batch=2):
    generator = torch.Generator().manual_seed(20260904)
    images = .2 + .55 * torch.rand((batch, 3, 28, 28), generator=generator)
    yy, xx = np.mgrid[-1:1:28j, -1:1:28j]
    weight = np.exp(-.5 * (((xx - .1) / .31)**2 + ((yy + .2) / .48)**2))
    return images, weight


class JointProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_active_subsets_match_and_backpropagate(self):
        images, weight = images_and_prf()
        for means in ((.47, None), (.47, .50)):
            for rms in ((None, None), (.18, None), (.18, .18)):
                for differentiable in (False, True):
                    with self.subTest(means=means, rms=rms, differentiable=differentiable):
                        x = images.clone().requires_grad_()
                        result = joint.match_luma_and_rms(
                            x, weight, target_global_mean=means[0], target_prf_mean=means[1],
                            target_global_rms=rms[0], target_prf_rms=rms[1],
                            differentiable=differentiable,
                        )
                        actual = joint._stats_torch(result.images.double(),
                            dual.normalized_prf_weight(weight, result.images.double())[0, 0])
                        for i, target in enumerate((*means, *rms)):
                            if target is not None:
                                self.assertLessEqual(float((actual[:, i] - target).abs().max().detach()) * 255, .01)
                        if differentiable:
                            grad = torch.autograd.grad(result.images.square().mean(), x)[0]
                            self.assertTrue(bool(grad.isfinite().all()))
                            self.assertGreater(float(grad.abs().sum()), 0)

    def test_parameters_reconstruct_one_transform_of_original(self):
        images, weight = images_and_prf(1)
        result = joint.match_luma_and_rms(images, weight, target_global_mean=116/255,
            target_prf_mean=122/255, target_global_rms=57.48/255, target_prf_rms=57.48/255)
        w = dual.normalized_prf_weight(weight, images)
        envelope = (w / w.max()).sqrt()
        mean_p, _ = dual.weighted_luma_mean_and_rms(images, w)
        local = (images + result.local_log_scale.expm1().view(-1,1,1,1)
                 * envelope * (images - mean_p.view(-1,1,1,1))
                 + result.local_luma_shift.view(-1,1,1,1) * envelope)
        mean_g, _ = dual.global_luma_mean_and_rms(local)
        mean_g = mean_g.view(-1,1,1,1)
        manual = (mean_g + result.global_luma_shift.view(-1,1,1,1)
                  + result.global_log_scale.exp().view(-1,1,1,1) * (local - mean_g)).clamp(0,1)
        torch.testing.assert_close(result.images, manual, atol=2e-6, rtol=0)
        metadata = joint.joint_result_to_jsonable(result)
        json.dumps(metadata, allow_nan=False)
        self.assertLessEqual(max(metadata["errors_byte"].values()), .01)

    def test_numeric_failures_retryable_but_configuration_errors_fatal(self):
        images, weight = images_and_prf(1)
        for value in (float('nan'), float('inf')):
            invalid = images.clone()
            invalid[0,0,0,0] = value
            with self.assertRaises(dual.DualScopeContrastConvergenceError) as raised:
                joint.match_luma_and_rms(invalid, weight, target_global_mean=.4)
            self.assertEqual(raised.exception.reason, "nonfinite_pixels")
        with self.assertRaises(ValueError):
            joint.match_luma_and_rms(images, weight, target_global_mean=2)
        with self.assertRaises(ValueError):
            joint.match_luma_and_rms(images, -weight, target_global_mean=.4)
        with patch('scipy.optimize.least_squares', side_effect=np.linalg.LinAlgError('SVD failed')):
            with self.assertRaises(dual.DualScopeContrastConvergenceError):
                joint.match_luma_and_rms(images, weight, target_global_mean=.4)
        # Uniform pRF cannot have a different mean than the whole image.
        with self.assertRaises(dual.DualScopeContrastConvergenceError):
            joint.match_luma_and_rms(images, np.ones((28,28)),
                                    target_global_mean=.3, target_prf_mean=.7)

    def test_guidance_stage_routing_and_disabled_luminance_regression(self):
        ns = pipeline_namespace()
        prepare = ns['prepare_brain_guidance_image']
        images, weight = images_and_prf(1)
        for contrast in ('none', 'final', 'every_step'):
            for luma in ('none', 'final', 'every_step'):
                kwargs = dict(contrast_constraint_method='match',
                    contrast_constraint_full_image=contrast, contrast_constraint_prf=contrast,
                    contrast_prf_weight=weight, target_rms_contrast=.18,
                    luminance_constraint_full_image=luma, luminance_constraint_prf=luma,
                    target_luminance_full_image=.47, target_luminance_prf=.50)
                output = prepare(images, **kwargs)
                stats = joint._stats_torch(output.double(),
                    dual.normalized_prf_weight(weight, output.double())[0,0])
                if luma == 'every_step':
                    self.assertLessEqual(float((stats[:,:2] - torch.tensor([.47,.50])).abs().max())*255, .01)
                if contrast == 'every_step':
                    self.assertLessEqual(float((stats[:,2:]-.18).abs().max())*255, .01)
                if luma != 'every_step' and contrast != 'every_step':
                    self.assertIs(output, images)
        output = prepare(images, contrast_constraint_method='match',
                         contrast_constraint_full_image='every_step', target_rms_contrast=.18)
        expected = ns['apply_rms_contrast_constraint'](images, method='match', target_contrast=.18)
        self.assertTrue(torch.equal(output, expected))

    def test_final_stage_reenforces_all_enabled_targets_and_retains_before(self):
        ns = pipeline_namespace()
        tree = source_tree('brain_guide_pipeline.py')
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'mypipelineSAG')
        call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
        start = next(i for i,n in enumerate(call.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.value) == 'self.decode_latents(latents)')
        end = next(i for i,n in enumerate(call.body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == 'run_safety_checker')
        images, weight = images_and_prf(1)
        raw = images.permute(0,2,3,1).numpy()
        for cstage, lstage in [('every_step','final'),('final','every_step'),('none','final'),('every_step','every_step')]:
            pipe = SimpleNamespace(decode_latents=lambda _: raw.copy(), last_unconstrained_image=None)
            scope = {**ns, 'self':pipe, 'latents':None, 'luma_contrast_method':'joint_solver',
                'apply_final_adjustment': True,
                'contrast_constraint_full_image':cstage, 'contrast_constraint_prf':cstage,
                'contrast_constraint_method':'match', 'contrast_prf_weight':weight,
                'luminance_constraint_full_image':lstage, 'luminance_constraint_prf':lstage,
                'target_rms_contrast':.18, 'target_luminance_full_image':.47,
                'target_luminance_prf':.50, 'dual_scope_solver_max_nfev':300,
                'dual_scope_tolerance':.01/255, 'luminance_validation_tolerance':.01/255,
                'dual_scope_parameter_limit':8., 'luminance_shift_limit':1.,
                'dual_scope_envelope_power':.5}
            run_nodes(call.body[start:end], scope)
            errors = pipe.last_joint_diagnostics['errors_byte']
            self.assertLessEqual(max(v for v in errors.values() if v is not None), .01)
            np.testing.assert_array_equal(pipe.last_unconstrained_image, raw)
            skip_scope = dict(scope, apply_final_adjustment=False)
            skip_scope.pop('match_luma_and_rms', None)
            pipe.last_joint_diagnostics = None
            run_nodes(call.body[start:end], skip_scope)
            np.testing.assert_array_equal(skip_scope['image'], raw)
            self.assertIsNone(pipe.last_joint_diagnostics)


class RunnerTests(unittest.TestCase):
    def test_cli_all_luminance_and_invalid_combinations(self):
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        def assignment_to(n, name):
            return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        start = next(i for i,n in enumerate(main.body) if assignment_to(n,'parser'))
        end = next(i for i,n in enumerate(main.body) if assignment_to(n,'percentile_values'))
        parse = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_ranking_top_n')
        resolve = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_luminance_targets')
        def execute(arguments):
            ns = dict(argparse=argparse, os=os, json=json, np=np, rois_folder='/test/rois',
                      build_seed_plan=build_seed_plan,
                      validate_luma_contrast_method=luma_clip.validate_luma_contrast_method,
                      method_metadata=luma_clip.method_metadata,
                      validate_luminance_settings=joint.validate_luminance_settings)
            run_nodes([parse, resolve],ns)
            with patch.object(sys,'argv',['runner','--luma_contrast_method','joint_solver',*arguments]), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                run_nodes(main.body[start:end],ns)
            return ns
        flags = ['--contrast_constraint_method','match',
                 '--luminance_constraint_full_image','final','--target_luminance_full_image','116',
                 '--luminance_constraint_prf','final','--target_luminance_prf','122',
                 '--ranking_top_n','all']
        ns = execute(flags)
        self.assertTrue(ns['luminance_options']['apply_final_adjustment'])
        raw_ns = execute(flags + ['--image_save_mode', 'unconstrained_only'])
        self.assertFalse(raw_ns['luminance_options']['apply_final_adjustment'])
        self.assertEqual(raw_ns['constraint_method_metadata']['effective_solver']['final'],
                         'skipped_unconstrained_only')
        self.assertEqual(ns['seed_plan']['mode'], 'random')
        fixed_flags = flags + ['--seed_mode', 'fixed', '--seed_generator_seed', '42']
        self.assertEqual(execute(fixed_flags)['seed_plan'], execute(fixed_flags)['seed_plan'])
        self.assertEqual(ns['args'].ranking_top_n,'all')
        self.assertAlmostEqual(ns['luminance_options']['target_luminance_prf'],122/255)
        self.assertEqual(ns['luminance_metadata']['acceptance_tolerance_byte'],.01)
        self.assertEqual(ns['luminance_metadata']['target_sources'],
                         {'full_image': 'explicit', 'prf': 'explicit'})
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / 'S2' / 'nsd_contrast_statistics.json'
            stats_path.parent.mkdir()
            stats_path.write_text(json.dumps({
                'subject': 2, 'image_count': 10,
                'luminance_mean': {'byte_0_255': {
                    'mean': 119.25, 'percentiles': {'50': 125}}}}))
            for stage in ('every_step', 'final'):
                automatic = ['--subject_id', '2', '--model_root', directory,
                             '--contrast_constraint_full_image', 'none',
                             '--luminance_constraint_full_image', stage,
                             '--luminance_constraint_prf', stage]
                ns = execute(automatic)
                for scope in ('full_image', 'prf'):
                    self.assertAlmostEqual(ns['luminance_options'][f'target_luminance_{scope}'], 119.25/255)
                self.assertEqual(ns['luminance_metadata']['statistics_path'], str(stats_path))
                # Explicit shared file works independently of model_root.
                ns = execute(automatic + ['--model_root', '/missing',
                             '--contrast_stats_json', str(stats_path),
                             '--target_luminance_prf', '121'])
                self.assertAlmostEqual(ns['luminance_options']['target_luminance_full_image'], 119.25/255)
                self.assertAlmostEqual(ns['luminance_options']['target_luminance_prf'], 121/255)
            for invalid_statistics in ({}, {'subject': 1},
                    {'luminance_mean': {'byte_0_255': {'mean': float('nan')}}},
                    {'luminance_mean': {'byte_0_255': {'mean': 256}}}):
                stats_path.write_text(json.dumps(invalid_statistics))
                with self.assertRaises(SystemExit):
                    execute(automatic)
            stats_path.unlink()
            with self.assertRaises(SystemExit):
                execute(automatic)
            # No file access when luminance is disabled or fully explicit.
            execute(['--model_root', directory])
            execute(flags + ['--model_root', directory])
        for invalid in (
            ['--luminance_constraint_full_image','final'],
            ['--luminance_constraint_prf','every_step','--target_luminance_prf','120'],
            ['--luminance_constraint_full_image','final','--target_luminance_full_image','120'],
            ['--ranking_top_n','0'],
            ['--seed_mode', 'fixed'],
            ['--seed_mode', 'file'],
            ['--seed_generator_seed', '42'],
        ):
            with self.assertRaises(SystemExit): execute(invalid)

    def test_mean_validation_does_not_enable_disabled_prf_contrast(self):
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='candidate_rejection_reason')
        image, weight = images_and_prf(1)
        result = joint.match_luma_and_rms(image, weight, target_global_mean=.47,
                                        target_prf_mean=.50, target_global_rms=.18)
        scope={**vars(dual), 'np':np, 'torch':torch,
            'args':SimpleNamespace(image_save_mode='both',luma_contrast_method='joint_solver',contrast_constraint_full_image='final',contrast_constraint_prf='none',
                contrast_constraint_method='match',luminance_constraint_full_image='final',
                luminance_constraint_prf='final',luminance_validation_tolerance=.01,target_rms_contrast=.18*255),
            'contrast_prf_weight':torch.as_tensor(weight), 'contrast_tolerance':.01/255,
            'luminance_options':dict(target_luminance_full_image=.47,target_luminance_prf=.50)}
        run_nodes([fn],scope)
        reason, metrics = scope['candidate_rejection_reason'](result.images[0].permute(1,2,0).numpy())
        self.assertIsNone(reason)
        self.assertLessEqual(metrics['global_mean_error_byte'],.01)
        self.assertLessEqual(metrics['prf_mean_error_byte'],.01)
        # Altering brightness by two byte units must reject before scoring.
        bad=result.images[0].permute(1,2,0).numpy().copy()
        bad=np.clip(bad+2/255,0,1)
        reason,_=scope['candidate_rejection_reason'](bad)
        self.assertEqual(reason,'global_luminance_mismatch')

    def test_all_parser_and_real_png_saving_over_999(self):
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        parser_def = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_ranking_top_n')
        scope={'argparse':argparse}
        run_nodes([parser_def], scope)
        parse = scope['parse_ranking_top_n']
        self.assertEqual(parse('all'), 'all')
        self.assertEqual(parse('30'), 30)
        for value in ('0','-1','2.5','bad'):
            with self.assertRaises(argparse.ArgumentTypeError): parse(value)
        # Execute actual ranking/saving block, including paired PNG filenames.
        voxel_loop = next(n for n in ast.walk(tree) if isinstance(n,ast.For)
            and any(isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ranked_indices'
                for t in s.targets) for s in n.body))
        start=next(i for i,n in enumerate(voxel_loop.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='ranked_indices' for t in n.targets))
        end=next(i for i,n in enumerate(voxel_loop.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='ranking_file' for t in n.targets))
        records=[dict(seed=i,tensor=torch.full((3,2,2),.5),raw_tensor=torch.full((3,2,2),.4),
            gen_score=-float(i),rank_score=float(i),rms_contrast_byte=1,raw_rms_contrast_byte=1,
            contrast_diagnostics={},dual_scope_solver=None,joint_luma_rms_solver=None,
            guidance_constraint_diagnostics={}) for i in range(1001)]
        for selection,count in [('all',1001),(30,30)]:
            with tempfile.TemporaryDirectory() as tmp:
                image_dir=os.path.join(tmp,'images'); raw_dir=os.path.join(tmp,'unconstrained_images')
                os.mkdir(image_dir); os.mkdir(raw_dir)
                scope=dict(image_records=records,args=SimpleNamespace(ranking_top_n=selection,enable_ranking='true'),
                    images_folder=image_dir,unconstrained_images_folder=raw_dir,os=os,np=np,Image=Image)
                run_nodes(voxel_loop.body[start:end], scope)
                self.assertEqual(len(scope['top_ranked']),count)
                self.assertEqual(len(os.listdir(image_dir)),count)
                self.assertEqual(sorted(os.listdir(image_dir)),sorted(os.listdir(raw_dir)))
                self.assertEqual(scope['top_ranked'][0]['seed'],1000)
                self.assertEqual(scope['top_ranked'][0]['gen_rank'],1001)
                self.assertEqual(scope['top_ranked'][0]['rank_model_rank'],1)
                self.assertEqual(scope['top_ranked'][0]['rank'],1)
                self.assertEqual(len(scope['candidate_rankings']),1001)
                self.assertEqual(sum(r['saved'] for r in scope['candidate_rankings']),count)
                self.assertEqual(scope['candidate_rankings'][0]['gen_rank'],1)
                self.assertEqual(scope['candidate_rankings'][0]['rank_model_rank'],1001)
                self.assertTrue(all(name.startswith('seed-') and 'rank-' not in name
                                    for name in os.listdir(image_dir)))
                if selection=='all':
                    self.assertTrue(os.path.exists(os.path.join(image_dir,'seed-0000000000.png')))

        # Exercise all three save modes with actual PNG writes and metadata paths.
        for mode in ('unconstrained_only', 'constrained_only', 'both'):
            with tempfile.TemporaryDirectory() as tmp:
                image_dir = os.path.join(tmp, 'images') if mode != 'unconstrained_only' else None
                raw_dir = os.path.join(tmp, 'unconstrained_images') if mode != 'constrained_only' else None
                for directory in (image_dir, raw_dir):
                    if directory is not None:
                        os.mkdir(directory)
                scope = dict(image_records=records[:3], args=SimpleNamespace(ranking_top_n=2,enable_ranking='true'),
                             images_folder=image_dir, unconstrained_images_folder=raw_dir,
                             os=os, np=np, Image=Image)
                run_nodes(voxel_loop.body[start:end], scope)
                self.assertEqual(len(scope['top_ranked']), 2)
                self.assertEqual(scope['top_ranked'][0]['seed'], 2)
                for item in scope['top_ranked']:
                    for path_key, directory in (('image_path', image_dir), ('raw_image_path', raw_dir)):
                        if directory is None:
                            self.assertIsNone(item[path_key])
                        else:
                            self.assertTrue(os.path.isfile(item[path_key]))
                            self.assertEqual(len(os.listdir(directory)), 2)
                self.assertEqual(os.path.isdir(os.path.join(tmp, 'images')), image_dir is not None)
                self.assertEqual(os.path.isdir(os.path.join(tmp, 'unconstrained_images')), raw_dir is not None)

        # Ranks are stable ordinal positions for ties; empty candidate lists save nothing.
        for subset in ([], [dict(r, gen_score=1., rank_score=1.) for r in records[:3]]):
            with tempfile.TemporaryDirectory() as tmp:
                scope = dict(image_records=subset, args=SimpleNamespace(ranking_top_n='all',enable_ranking='true'),
                             images_folder=tmp, unconstrained_images_folder=None,
                             os=os, np=np, Image=Image)
                run_nodes(voxel_loop.body[start:end], scope)
                self.assertEqual([r['gen_rank'] for r in scope['top_ranked']], list(range(1,len(subset)+1)))
                self.assertEqual([r['rank_model_rank'] for r in scope['top_ranked']], list(range(1,len(subset)+1)))

    def test_rejected_seed_is_replaced_without_losing_accepted_images(self):
        tree=source_tree('run_image_generation_concat_pRF_model_ranking.py')
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate_pixel_valid_candidates')
        class FakePipeline:
            calls=0
            last_unconstrained_image=None
            last_contrast_diagnostics=None
            last_joint_diagnostics=None
            last_guidance_constraint_diagnostics={}
            def __call__(self,*args,**kwargs):
                self.calls+=1
                if self.calls==2:
                    dual.require_finite_seed_image(torch.tensor(float('nan')),stage='test')
                return SimpleNamespace(images=[np.full((4,4,3),.5,dtype=np.float32)])
        args=SimpleNamespace(luma_contrast_method='joint_solver',num_seeds=2,contrast_constraint_method='match',
            contrast_constraint_full_image='final',contrast_constraint_prf='final',
            min_rms_contrast=None,max_rms_contrast=None,target_rms_contrast=57.48,
            dual_scope_solver_max_nfev=300,dual_scope_every_step_iterations=10,
            dual_scope_parameter_limit=8.,dual_scope_envelope_power=.5)
        clock_ticks = iter(range(30))
        scope=dict(args=args,pipe=FakePipeline(),torch=torch,np=np,
            time=SimpleNamespace(perf_counter=lambda: next(clock_ticks)), random=random.Random(7),
            image_records=[],attempted_seeds=[],max_generation_attempts=4,device=torch.device('cpu'),
            seed_generation_times=[],
            seed_plan={'seed_set': [101, 202, 303, 404]}, checkpoint_seeds=lambda: None,
            sag_scale=.75,num_steps=3,brain_guidance_scale=30,contrast_prf_weight=None,
            contrast_tolerance=.01/255,luminance_options={},generation_timing={'seconds':0.,'calls':0},
            rejected_candidates={'nonfinite_pixels':0},
            DualScopeContrastConvergenceError=dual.DualScopeContrastConvergenceError,
            unconstrained_candidate_image=lambda _: (None,None),
            candidate_rejection_reason=lambda _: (None,{'float_global_rms_byte':57.48}))
        run_nodes([fn],scope)
        with contextlib.redirect_stdout(io.StringIO()):
            records=scope['generate_pixel_valid_candidates'](2)
        self.assertEqual(len(records),2)
        self.assertEqual(scope['pipe'].calls,3)
        self.assertEqual(scope['rejected_candidates']['nonfinite_pixels'],1)
        self.assertEqual(len(scope['attempted_seeds']),3)
        self.assertEqual(scope['attempted_seeds'], [101, 202, 303])
        self.assertEqual([item['seed'] for item in records], [101, 303])
        self.assertEqual(scope['generation_timing'], {'seconds': 3., 'calls': 3})
        self.assertEqual([item['generation_seconds'] for item in scope['seed_generation_times']], [1., 1., 1.])
        self.assertEqual([item['pipeline_returned'] for item in scope['seed_generation_times']], [True, False, True])
        # A file can have fewer entries than num_seeds: stop, never invent seeds.
        scope.update(pipe=FakePipeline(), attempted_seeds=[], max_generation_attempts=2,
                     seed_plan={'seed_set': [11, 22]})
        with contextlib.redirect_stdout(io.StringIO()):
            records = scope['generate_pixel_valid_candidates'](2)
        self.assertEqual([item['seed'] for item in records], [11])
        self.assertEqual(scope['attempted_seeds'], [11, 22])

    def test_seed_audit_excludes_nonfinite_scores(self):
        from seed_utils import build_seed_plan, save_seed_record
        from generation_timing_utils import merge_roi_generation_timing
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        score_loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Tuple)
                         and [v.id for v in n.target.elts if isinstance(v, ast.Name)]
                         == ['item', 'gen_score', 'rank_score'])
        checkpoint = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == 'checkpoint_seeds')
        with tempfile.TemporaryDirectory() as directory:
            plan = build_seed_plan('fixed', 3, generator_seed=10)
            attempts = plan['seed_set'][:3]
            path = os.path.join(directory, 'seeds.json')
            scope = dict(np=np, image_records=[], attempted_seeds=attempts,
                         seed_plan=plan, seed_record_path=path, save_seed_record=save_seed_record,
                         pending_records=[dict(seed=s, tensor=None, raw_tensor=None) for s in attempts],
                         gen_scores=[1., float('nan'), 1.], pred_scores=[2., 2., float('inf')],
                         rejected_candidates={'nonfinite_gen_score': 0, 'nonfinite_rank_score': 0})
            scope.update(os=os, ss=1, region='V1', voxel_id=5, experiment_folder=directory,
                         region_folder=os.path.join(directory, 'roi-V1'),
                         merge_roi_generation_timing=merge_roi_generation_timing,
                         hardware_metadata={'device_type': 'cuda', 'gpu_name': 'Mock GPU'},
                         args=SimpleNamespace(generation_timing_file=None,enable_ranking='true'),
                         seed_generation_times=[dict(seed=s, generation_seconds=float(i+1), pipeline_returned=True)
                                                for i, s in enumerate(attempts)],
                         generation_timing={'seconds': 6., 'calls': 3, 'voxels': {}},
                         save_json_atomic=seed_utils.save_json_atomic,
                         summarize_generation_times=seed_utils.summarize_generation_times,
                         GENERATION_TIMING_SCOPE=seed_utils.GENERATION_TIMING_SCOPE)
            run_nodes([checkpoint], scope)
            with contextlib.redirect_stdout(io.StringIO()):
                run_nodes([score_loop], scope)
            scope['checkpoint_seeds']()
            record = json.loads(Path(path).read_text())
            self.assertEqual(record['all_seeds_attempted'], attempts)
            self.assertEqual(record['valid_seeds'], attempts[:1])
            self.assertEqual(record['generation_timing']['generation_seconds'], 6.)
            self.assertEqual([r['valid_seed'] for r in record['generation_timing']['per_seed']], [True, False, False])
            total = json.loads((Path(directory) / 'roi-V1' / 'generation_timing.json').read_text())
            self.assertEqual(total['generation_seconds'], 6.)
            self.assertEqual(total['seconds_per_generation'], 2.)
            self.assertEqual(total['per_voxel'][0]['seed_record'], path)
            self.assertEqual(total['hardware'], scope['hardware_metadata'])
            self.assertNotIn('hardware', record)
            # Repeated checkpoints replace a voxel summary, not double-count it.
            scope['checkpoint_seeds']()
            second_path = os.path.join(directory, 'voxel2', 'seeds.json')
            mirror = os.path.join(directory, 'timing_copy.json')
            scope.update(voxel_id=6, seed_record_path=second_path, attempted_seeds=attempts[:1],
                         image_records=[], seed_generation_times=[dict(seed=attempts[0], generation_seconds=4.)],
                         args=SimpleNamespace(generation_timing_file=mirror,enable_ranking='true'))
            scope['generation_timing'].update(seconds=10., calls=4)
            scope['checkpoint_seeds']()
            total = json.loads((Path(directory) / 'roi-V1' / 'generation_timing.json').read_text())
            self.assertEqual(total['generation_seconds'], 10.)
            self.assertEqual(total['generation_calls'], 4)
            self.assertEqual(len(total['per_voxel']), 2)
            self.assertEqual(sum(v['generation_seconds'] for v in total['per_voxel']), 10.)
            mirror_data = json.loads((Path(directory) / 'roi-V1' / 'timing_copy.json').read_text())
            self.assertEqual(mirror_data['generation_seconds'], 4.)
            self.assertFalse(Path(mirror).exists())
            self.assertFalse((Path(directory) / 'generation_timing.json').exists())
            # A new ROI in this process must not inherit V1's accumulated time.
            scope.update(region='V2', region_folder=os.path.join(directory, 'roi-V2'),
                         seed_record_path=os.path.join(directory, 'roi-V2', 'voxel3', 'seeds.json'))
            scope['checkpoint_seeds']()
            v2 = json.loads((Path(directory) / 'roi-V2' / 'generation_timing.json').read_text())
            self.assertEqual(v2['generation_seconds'], 4.)
            self.assertEqual(v2['generation_calls'], 1)
            self.assertEqual(v2['region'], 'V2')
            self.assertEqual(json.loads((Path(directory) / 'roi-V1' / 'generation_timing.json').read_text()), total)


if __name__ == '__main__':
    unittest.main()
