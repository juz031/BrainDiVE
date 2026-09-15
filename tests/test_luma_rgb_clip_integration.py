"""CPU tests for the luma-only method and actual runner/pipeline wiring."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import luma_rgb_clip_utils as new
import dual_scope_contrast_utils as dual
from seed_utils import build_seed_plan
from tests.test_joint_luma_integration import (ROOT, images_and_prf, pipeline_namespace,
                                              run_nodes, source_tree)


class LumaClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_only_matches_read_only_reference(self):
        path = Path('/home/hanfeig/gradient_descent_image_generation/contrast_constraints/luma_mean_rms_rgb_clip.py')
        if not path.exists():
            self.skipTest('Read-only external reference not available')
        # Execute the reference source without creating bytecode in its directory.
        ref = {}
        exec(compile(path.read_text(), str(path), 'exec'), ref)
        for seed in (1, 42, 99):
            x = torch.rand((2, 3, 32, 32), generator=torch.Generator().manual_seed(seed))
            expected, diag = ref['project_luma_mean_rms_then_rgb_clip'](
                x, target_mean=116/255, target_rms=57.48/255, scalar_tolerance=1/255)
            actual, ours = new.match_luma_rgb_clip(x, target_global_mean=116/255,
                target_global_rms=57.48/255, mean_tolerance=1/255, rms_tolerance=1/255)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
            np.testing.assert_allclose(ours['postclip_errors_byte']['global_rms'],
                                       diag['postclip_rms_error'].numpy()*255, atol=2e-4)

    def test_dual_scope_matches_luma_and_restores_original_color(self):
        images, weight = images_and_prf(1)
        y = (images * images.new_tensor(new.REC709).view(1,3,1,1)).sum(1, keepdim=True)
        chroma = images - y
        for differentiable in (False, True):
            x = images.clone().requires_grad_()
            output, diag = new.match_luma_rgb_clip(x, weight,
                target_global_mean=.47, target_prf_mean=.50,
                target_global_rms=.18, target_prf_rms=.18,
                differentiable=differentiable, iterations=30)
            self.assertEqual(diag['chroma_alpha'], 1)
            for errors in diag['preclip_errors_byte'].values():
                self.assertLessEqual(max(errors), .01 + 1e-5)
            # Chroma differences remain original wherever reconstructed channels
            # have not clipped. This would fail for an RGB contrast-scaling fallback.
            mask = ((output > 1e-5) & (output < 1-1e-5)).all(1)
            torch.testing.assert_close((output[:,0]-output[:,1])[mask],
                                       (images[:,0]-images[:,1])[mask], atol=2e-6, rtol=0)
            self.assertGreater(int(mask.sum()), 0)
            if differentiable:
                grad = torch.autograd.grad(output.square().mean(), x)[0]
                self.assertTrue(bool(grad.isfinite().all()))
                self.assertGreater(float(grad.abs().sum()), 0)
            json.dumps(diag, allow_nan=False)

    def test_full_only_gradient_and_postclip_error_not_rejected(self):
        x = torch.rand((1,3,32,32), generator=torch.Generator().manual_seed(42), requires_grad=True)
        output, diag = new.match_luma_rgb_clip(x, target_global_mean=116/255,
            target_global_rms=57.48/255, mean_tolerance=.01/255, rms_tolerance=.01/255)
        self.assertGreater(diag['postclip_errors_byte']['global_rms'][0], .01)
        self.assertLessEqual(diag['preclip_errors_byte']['global_rms'][0], .01)
        self.assertEqual(diag['postclip_mismatch_policy'], 'record_only')
        grad = torch.autograd.grad(output.square().mean(), x)[0]
        self.assertTrue(bool(grad.isfinite().all()))

    def test_numeric_failures_retryable_and_invalid_configs_fatal(self):
        with self.assertRaises(dual.DualScopeContrastConvergenceError) as cm:
            new.match_luma_rgb_clip(torch.full((1,3,8,8), .5),
                target_global_mean=.5, target_global_rms=.2, match_iterations=1)
        self.assertEqual(cm.exception.reason, 'luma_rgb_clip_solver_failure')
        with self.assertRaises(dual.DualScopeContrastConvergenceError) as cm:
            new.match_luma_rgb_clip(torch.full((1,3,8,8), float('nan')),
                target_global_mean=.5, target_global_rms=.2)
        self.assertEqual(cm.exception.reason, 'nonfinite_pixels')
        images, _ = images_and_prf(1)
        with self.assertRaises(dual.DualScopeContrastConvergenceError) as cm:
            new.match_luma_rgb_clip(images, torch.ones((28,28)),
                target_global_mean=.3, target_prf_mean=.7,
                target_global_rms=.18, target_prf_rms=.18)
        self.assertEqual(cm.exception.reason, 'luma_rgb_clip_solver_failure')
        with self.assertRaises(ValueError):
            new.match_luma_rgb_clip(images, target_global_mean=.5,
                target_global_rms=.2, target_prf_mean=.4)

    def test_all_timing_combinations_and_guidance_routing(self):
        ns = pipeline_namespace()
        prepare = ns['prepare_brain_guidance_image']
        images, weight = images_and_prf(1)
        for full, prf in [('none','none'), ('final','none'), ('final','final'),
                          ('every_step','none'), ('every_step','final'), ('every_step','every_step')]:
            new.validate_luma_contrast_method('luma_rgb_clip',full,prf,full,prf,'match',20)
            summary = {}
            output = prepare(images, luma_contrast_method='luma_rgb_clip',
                contrast_constraint_method='match', contrast_constraint_full_image=full,
                contrast_constraint_prf=prf, luminance_constraint_full_image=full,
                luminance_constraint_prf=prf, contrast_prf_weight=weight,
                target_rms_contrast=.18, target_luminance_full_image=.47,
                target_luminance_prf=.50, dual_scope_every_step_iterations=30,
                guidance_constraint_diagnostics=summary)
            meta = new.method_metadata('luma_rgb_clip',full,prf,20)
            if full != 'every_step':
                self.assertIs(output, images)
                self.assertEqual(summary, {})
            else:
                self.assertEqual(summary['evaluations'], 1)
                self.assertEqual(summary['last']['backend'], meta['effective_solver']['every_step'])
        for stages in [('every_step','none','final','none'), ('final','every_step','final','every_step')]:
            with self.assertRaises(ValueError):
                new.validate_luma_contrast_method('luma_rgb_clip',*stages,'match',20)
        for method in ('max','range'):
            with self.assertRaises(ValueError):
                new.validate_luma_contrast_method('luma_rgb_clip','final','none','final','none',method,20)

    def test_actual_final_pipeline_routes_to_new_method(self):
        ns = pipeline_namespace()
        tree = source_tree('brain_guide_pipeline.py')
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='mypipelineSAG')
        call = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
        start = next(i for i,n in enumerate(call.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.value)=='self.decode_latents(latents)')
        end = next(i for i,n in enumerate(call.body) if isinstance(n,ast.Assign)
                   and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                   and n.value.func.attr=='run_safety_checker')
        images,weight = images_and_prf(1)
        raw = images.permute(0,2,3,1).numpy()
        for full,prf in [('final','none'),('final','final'),('every_step','none'),
                         ('every_step','final'),('every_step','every_step')]:
            pipe = SimpleNamespace(decode_latents=lambda _:raw.copy(),last_unconstrained_image=None)
            scope = dict(ns, self=pipe, latents=None, luma_contrast_method='luma_rgb_clip',
                apply_final_adjustment=True,
                luma_rgb_clip_match_iterations=20,
                contrast_constraint_full_image=full, contrast_constraint_prf=prf,
                luminance_constraint_full_image=full,luminance_constraint_prf=prf,
                contrast_constraint_method='match',contrast_prf_weight=weight,
                target_rms_contrast=.18,target_luminance_full_image=.47,target_luminance_prf=.50,
                dual_scope_solver_max_nfev=300,dual_scope_tolerance=.01/255,
                luminance_validation_tolerance=.01/255,dual_scope_parameter_limit=8.,
                luminance_shift_limit=1.,dual_scope_envelope_power=.5)
            run_nodes(call.body[start:end], scope)
            np.testing.assert_array_equal(pipe.last_unconstrained_image, raw)
            diag = pipe.last_joint_diagnostics
            self.assertEqual(diag['method'],'luma_rgb_clip')
            self.assertEqual(diag['backend'],'reference_full_image_luma_loop' if prf=='none' else 'coupled_luma_scipy')
            self.assertEqual(diag['acceptance_domain'],new.DOMAIN)
            # Skipping must bypass the solver, not just ignore its output/error.
            skip_scope = dict(scope, apply_final_adjustment=False)
            for solver in ('match_luma_rgb_clip', 'match_luma_and_rms',
                           'match_global_and_prf_rms_contrast_final', 'apply_rms_contrast_constraint'):
                skip_scope.pop(solver, None)
            pipe.last_joint_diagnostics = None
            run_nodes(call.body[start:end], skip_scope)
            np.testing.assert_array_equal(skip_scope['image'], raw)
            np.testing.assert_array_equal(pipe.last_unconstrained_image, raw)
            self.assertIsNone(pipe.last_joint_diagnostics)

    def test_runner_acceptance_domain_and_pixel_checks(self):
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='candidate_rejection_reason')
        images,weight = images_and_prf(1)
        args = SimpleNamespace(image_save_mode='both',luma_contrast_method='luma_rgb_clip',contrast_constraint_full_image='final',
            contrast_constraint_prf='final',luminance_constraint_full_image='final',luminance_constraint_prf='final',
            contrast_constraint_method='match',luminance_validation_tolerance=.01,target_rms_contrast=.18*255)
        ns = dict(vars(dual),np=np,torch=torch,args=args,contrast_prf_weight=torch.as_tensor(weight),
                  contrast_tolerance=.01/255,acceptance_domain=new.DOMAIN,
                  luminance_options=dict(target_luminance_full_image=.47,target_luminance_prf=.50))
        run_nodes([fn],ns)
        array = images[0].permute(1,2,0).numpy()
        reason,metrics = ns['candidate_rejection_reason'](array)
        self.assertIsNone(reason)  # Output mismatches are diagnostic for new method.
        self.assertEqual(metrics['postclip_mismatch_policy'],'record_only')
        args.luma_contrast_method='joint_solver'
        self.assertIsNotNone(ns['candidate_rejection_reason'](array)[0])
        args.image_save_mode='unconstrained_only'
        for method in ('joint_solver', 'luma_rgb_clip'):
            args.luma_contrast_method=method
            reason, metrics = ns['candidate_rejection_reason'](array)
            self.assertIsNone(reason)
            self.assertFalse(metrics['final_target_validation_applied'])
        args.luma_contrast_method='luma_rgb_clip'
        for bad,expected in [(array[:,:,:2],'invalid_shape'), (array*0+np.nan,'nonfinite_pixels'),
                             (array*0+2,'out_of_range_pixels')]:
            self.assertEqual(ns['candidate_rejection_reason'](bad)[0], expected)

    def test_cli_new_default_and_validation(self):
        tree = source_tree('run_image_generation_concat_pRF_model_ranking.py')
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        named = lambda n,name: isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
        start = next(i for i,n in enumerate(main.body) if named(n,'parser'))
        end = next(i for i,n in enumerate(main.body) if named(n,'percentile_values'))
        helpers = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('parse_ranking_top_n','resolve_luminance_targets')]
        def execute(flags):
            ns = dict(vars(new),argparse=argparse,os=os,np=np,json=json,rois_folder='/test',
                      build_seed_plan=build_seed_plan,
                      validate_luminance_settings=__import__('joint_luma_contrast_utils').validate_luminance_settings)
            run_nodes(helpers,ns)
            with patch.object(sys,'argv',['runner',*flags]),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                run_nodes(main.body[start:end],ns)
            return ns
        flags=['--contrast_constraint_method','match','--luminance_constraint_full_image','every_step',
               '--target_luminance_full_image','116','--target_luminance_prf','116']
        ns=execute(flags)
        self.assertEqual(ns['args'].luma_contrast_method,'luma_rgb_clip')
        self.assertEqual(ns['luminance_options']['luma_rgb_clip_match_iterations'],20)
        self.assertEqual(ns['acceptance_domain'],new.DOMAIN)
        with self.assertRaises(SystemExit):
            execute(flags+['--luminance_constraint_full_image','final'])
        execute(['--contrast_constraint_full_image','none'])  # all off

    def test_projection_failure_retries_the_next_seed(self):
        tree=source_tree('run_image_generation_concat_pRF_model_ranking.py')
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate_pixel_valid_candidates')
        class FakePipeline:
            last_unconstrained_image=None
            last_contrast_diagnostics=None
            last_joint_diagnostics=None
            last_guidance_constraint_diagnostics={}
            calls=0
            def __call__(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    new.match_luma_rgb_clip(torch.full((1,3,8,8),.5),
                        target_global_mean=.5,target_global_rms=.2,match_iterations=1)
                return SimpleNamespace(images=[np.full((4,4,3),.5,dtype=np.float32)])
        args=SimpleNamespace(luma_contrast_method='luma_rgb_clip',num_seeds=1,
            contrast_constraint_method='match',contrast_constraint_full_image='final',
            contrast_constraint_prf='none',min_rms_contrast=None,max_rms_contrast=None,
            target_rms_contrast=57.48,dual_scope_solver_max_nfev=300,
            dual_scope_every_step_iterations=10,dual_scope_parameter_limit=8.,dual_scope_envelope_power=.5)
        ticks=iter(range(20))
        ns=dict(args=args,pipe=FakePipeline(),torch=torch,np=np,
            time=SimpleNamespace(perf_counter=lambda:next(ticks)),image_records=[],attempted_seeds=[],
            max_generation_attempts=2,device=torch.device('cpu'),seed_generation_times=[],
            seed_plan={'seed_set':[101,202]},checkpoint_seeds=lambda:None,
            sag_scale=.75,num_steps=3,brain_guidance_scale=30,contrast_prf_weight=None,
            contrast_tolerance=.01/255,luminance_options={},generation_timing={'seconds':0.,'calls':0},
            DualScopeContrastConvergenceError=dual.DualScopeContrastConvergenceError,
            rejected_candidates={'luma_rgb_clip_solver_failure':0},
            unconstrained_candidate_image=lambda _:(None,None),
            candidate_rejection_reason=lambda _:(None,{'float_global_rms_byte':57.48}))
        run_nodes([fn],ns)
        with contextlib.redirect_stdout(io.StringIO()):
            records=ns['generate_pixel_valid_candidates'](1)
        self.assertEqual([r['seed'] for r in records],[202])
        self.assertEqual(ns['rejected_candidates']['luma_rgb_clip_solver_failure'],1)
        self.assertEqual(ns['attempted_seeds'],[101,202])

    def test_actual_output_paths_separate_methods(self):
        tree=source_tree('run_image_generation_concat_pRF_model_ranking.py')
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in ('settings_folder','experiment_folder') for t in n.targets)]
        paths=[]
        for method in ('luma_rgb_clip','joint_solver'):
            args=SimpleNamespace(split_id=1,enable_ranking='true',contrast_constraint_full_image='every_step',
                contrast_constraint_prf='final',num_seeds=1000,ranking_top_n='all',
                luma_contrast_method=method,save_root='/tmp/report-test',subject_id=1,seed_mode='fixed')
            ns=dict(args=args,num_steps=100,brain_guidance_scale=250,sag_scale=.75,
                contrast_folder='match-rms57.48',luma_full='every_step-116',luma_prf='final-116',
                os=os,model_folder='gen-DINO',seed_plan={'plan_id':'test'})
            run_nodes(nodes,ns)
            path=Path(ns['experiment_folder']);paths.append(path)
            self.assertIn('method-'+method,path.parts)
            self.assertTrue(all(len(p)<256 for p in path.parts))
        self.assertNotEqual(paths[0],paths[1])


if __name__ == '__main__':
    unittest.main()
