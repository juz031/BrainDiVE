from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mei_analysis.features.basic import weighted_mean_std
from mei_analysis.features.gabor import pool_response_maps
from mei_analysis.manifest import build_work_units, discover_contexts
from mei_analysis.prfs import generation_prf
from mei_analysis.utils import load_config, subject_path


CONFIG = ROOT / "config" / "analysis_s1.yaml"


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)

    def test_weighted_constant(self):
        values = np.full((8, 8), 7.25)
        weights = np.arange(1, 65, dtype=float).reshape(8, 8)
        mean, std = weighted_mean_std(values, weights)
        self.assertAlmostEqual(mean, 7.25)
        self.assertAlmostEqual(std, 0.0)

    def test_gabor_pooling_happens_on_response_maps(self):
        response = torch.zeros((1, 4, 4, 2), dtype=torch.float32)
        response[:, 1, 2, :] = torch.tensor([4.0, 8.0])
        prf = torch.zeros((4, 4), dtype=torch.float32)
        prf[1, 2] = 1.0
        full, weighted = pool_response_maps(response, prf)
        np.testing.assert_allclose(full.numpy(), [[0.25, 0.5]])
        np.testing.assert_allclose(weighted.numpy(), [[4.0, 8.0]])

    def test_current_work_unit_count_and_exclusions(self):
        contexts = discover_contexts(self.config)
        units = build_work_units(self.config, contexts)
        self.assertEqual(len(contexts), 160)
        self.assertEqual(len(units), 960)
        self.assertNotIn("gradient_pixel_raw", set(units["condition"]))
        self.assertNotIn("gradient_maco_raw", set(units["condition"]))

    def test_generation_prf_is_normalized(self):
        contexts = discover_contexts(self.config)
        row = contexts.iloc[0]
        prf_id, params, mask = generation_prf(
            self.config, row.subject, row.backbone, int(row.voxel_id)
        )
        self.assertGreaterEqual(prf_id, 0)
        self.assertEqual(params.shape, (3,))
        self.assertEqual(mask.shape, (256, 256))
        self.assertAlmostEqual(float(mask.sum()), 1.0, places=6)

    def test_subject_path_templates(self):
        self.assertEqual(subject_path("/x/sub-{subject_number_02}", "S1"), Path("/x/sub-01"))
        self.assertEqual(subject_path("/x/{subject}", "S1"), Path("/x/S1"))


if __name__ == "__main__":
    unittest.main()
