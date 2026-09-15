import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from seed_utils import build_seed_plan, save_seed_record, collect_device_metadata


class SeedPlanTests(unittest.TestCase):
    def test_device_metadata_uses_selected_gpu(self):
        properties = SimpleNamespace(name='Mock GPU', total_memory=24 * 2**30,
                                     major=8, minor=6, uuid='GPU-test')
        cuda = SimpleNamespace(current_device=Mock(return_value=1),
                               get_device_properties=Mock(return_value=properties))
        torch_stub = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'), cuda=cuda)
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3,5', 'SLURM_JOB_ID': '123'}):
            for index, expected in ((None, 1), (0, 0)):
                result = collect_device_metadata(SimpleNamespace(type='cuda', index=index), torch_stub)
                cuda.get_device_properties.assert_called_with(expected)
                self.assertEqual(result['cuda_device_index'], expected)
                self.assertEqual(result['gpu_name'], 'Mock GPU')
                self.assertEqual(result['gpu_uuid'], 'GPU-test')
                self.assertEqual(result['total_memory_gib'], 24)
                self.assertEqual(result['cuda_visible_devices'], '3,5')
                self.assertEqual(result['slurm_job_id'], '123')
                json.dumps(result, allow_nan=False)

    def test_cpu_metadata_and_optional_uuid(self):
        torch_stub = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None))
        result = collect_device_metadata(SimpleNamespace(type='cpu', index=None), torch_stub)
        self.assertIsNone(result['gpu_name'])
        properties = SimpleNamespace(name='Mock GPU', total_memory=1024, major=7, minor=0)
        torch_stub.cuda = SimpleNamespace(get_device_properties=lambda _: properties)
        result = collect_device_metadata(SimpleNamespace(type='cuda', index=0), torch_stub)
        self.assertIsNone(result['gpu_uuid'])

    def test_fixed_reproducible_and_prefix_stable(self):
        a = build_seed_plan('fixed', 3, generator_seed=42)
        b = build_seed_plan('fixed', 3, generator_seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a['seed_set']), 13)
        longer = build_seed_plan('fixed', 30, generator_seed=42)
        self.assertEqual(a['seed_set'], longer['seed_set'][:13])
        self.assertNotEqual(a['seed_set'], build_seed_plan('fixed', 3, generator_seed=43)['seed_set'])

    def test_random_unique_and_independent(self):
        a = build_seed_plan('random', 100)
        b = build_seed_plan('random', 100)
        self.assertEqual(len(set(a['seed_set'])), 110)
        self.assertNotEqual(a['plan_id'], b['plan_id'])
        self.assertIsNone(a['seed_generator_seed'])

    def test_save_and_reload_each_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seeds.json'
            plan = build_seed_plan('fixed', 3, generator_seed=7)
            attempts = plan['seed_set'][:4]
            valid = attempts[1:]
            save_seed_record(path, plan, attempts, valid, 'complete')
            data = json.loads(path.read_text())
            self.assertEqual(data['all_seeds_attempted'], attempts)
            self.assertEqual(data['valid_seeds'], valid)
            self.assertEqual(data['seed_generator_seed'], 7)
            for key in ('all_seeds_attempted', 'valid_seeds', 'seed_set'):
                loaded = build_seed_plan('file', 3, seed_file=path, file_key=key)
                self.assertEqual(loaded['seed_set'], data[key])
                self.assertEqual(loaded['seed_file'], str(path))
                self.assertEqual(len(loaded['seed_file_sha256']), 64)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_short_file_and_attempt_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text('[19, 3, 0]')
            plan = build_seed_plan('file', 10, seed_file=path)
            self.assertEqual(plan['seed_set'], [19, 3, 0])
            self.assertEqual(plan['effective_max_attempts'], 3)
            plan = build_seed_plan('file', 1, max_attempts=2, seed_file=path)
            self.assertEqual(plan['effective_max_attempts'], 2)
            self.assertEqual(plan['seed_set'], [19, 3, 0])

    def test_bad_configuration(self):
        for mode, kwargs in [
            ('fixed', {}), ('file', {}), ('bad', {}),
            ('random', {'generator_seed': 5}), ('fixed', {'generator_seed': 5, 'seed_file': 'x'}),
            ('random', {'max_attempts': 1}),
        ]:
            with self.subTest(mode=mode, kwargs=kwargs), self.assertRaises(ValueError):
                build_seed_plan(mode, 3, **kwargs)

    def test_invalid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            for data in ([], [1, 1], [-1], [2**32], [True], [1.5], ['1'], {}, None):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises((ValueError, KeyError, TypeError)):
                    build_seed_plan('file', 1, seed_file=path)

    def test_checkpoint_updates_do_not_mutate_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'voxel' / 'seeds.json'
            plan = build_seed_plan('fixed', 1, generator_seed=1)
            first, second = plan['seed_set'][:2]
            save_seed_record(path, plan, [first], [])
            self.assertEqual(json.loads(path.read_text())['status'], 'in_progress')
            save_seed_record(path, plan, [first, second], [second], 'complete')
            result = json.loads(path.read_text())
            self.assertEqual(result['all_seeds_attempted'], [first, second])
            self.assertEqual(result['valid_seeds'], [second])
            self.assertNotIn('all_seeds_attempted', plan)


if __name__ == '__main__':
    unittest.main()
