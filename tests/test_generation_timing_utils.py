"""ROI timing isolation and concurrent checkpoint regression tests."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from generation_timing_utils import merge_roi_generation_timing


def worker(path, index):
    row = dict(region='V1', voxel_id=index, seed_record=f'{path}/../voxel-{index}/seeds.json',
               generation_seconds=2., generation_calls=1, hardware={'gpu_name': f'GPU-{index % 2}'})
    merge_roi_generation_timing(path, 'V1', [row])
    row.update(generation_seconds=5., generation_calls=2)
    merge_roi_generation_timing(path, 'V1', [row])


class TimingTests(unittest.TestCase):
    def test_concurrent_workers_preserve_voxels_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'roi-V1' / 'generation_timing.json')
            with ProcessPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker, [path] * 12, range(12)))
            data = json.loads(Path(path).read_text())
            self.assertEqual(len(data['per_voxel']), 12)
            self.assertEqual(data['generation_seconds'], 60.)
            self.assertEqual(data['generation_calls'], 24)
            self.assertEqual(data['seconds_per_generation'], 2.5)
            self.assertIsNone(data['hardware'])
            self.assertTrue(all(row['hardware'] for row in data['per_voxel']))
            # A restarted process replaces its voxel snapshot instead of adding it.
            worker(path, 0)
            self.assertEqual(json.loads(Path(path).read_text())['generation_seconds'], 60.)

    def test_reject_wrong_roi_without_changing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'generation_timing.json')
            worker(path, 1)
            original = Path(path).read_bytes()
            with self.assertRaises(ValueError):
                merge_roi_generation_timing(path, 'V2', [])
            self.assertEqual(Path(path).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
