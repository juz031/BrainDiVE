import unittest

import numpy as np
import torch

from max_activation_utils import RMSContrastConstraint
from scripts.validate_fixed_prf_candidates import (
    compare_with_natural_pool,
    contrast_match_natural_batch,
)


class ContrastMatchedValidationTests(unittest.TestCase):
    def test_matching_uses_luminance_mean_and_rms_after_8bit_quantization(self):
        rng = np.random.default_rng(13)
        pixels = rng.integers(
            0,
            256,
            size=(4, 3, 48, 48),
            dtype=np.uint8,
        )
        constraint = RMSContrastConstraint(
            target=16.0 / 255.0,
            mode="match",
            target_mean=111.0 / 255.0,
        )

        matched, means, rms = contrast_match_natural_batch(
            pixels,
            constraint,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(matched.shape), (4, 3, 48, 48))
        self.assertTrue(torch.all(matched >= 0.0))
        self.assertTrue(torch.all(matched <= 1.0))
        np.testing.assert_allclose(means, 111.0, atol=0.02)
        np.testing.assert_allclose(rms, 16.0, atol=0.02)

    def test_hwc_natural_storage_is_supported(self):
        pixels = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        pixels[:, :, 8:, :] = 255
        constraint = RMSContrastConstraint(
            target=16.0 / 255.0,
            mode="match",
            target_mean=111.0 / 255.0,
        )

        matched, means, rms = contrast_match_natural_batch(
            pixels,
            constraint,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(matched.shape), (2, 3, 16, 16))
        np.testing.assert_allclose(means, 111.0, atol=0.02)
        np.testing.assert_allclose(rms, 16.0, atol=0.02)

    def test_raw_and_matched_natural_populations_are_ranked_separately(self):
        candidates = [
            {"condition": "contrast_only", "seed": 1, "image_path": "one.png"},
            {"condition": "smooth_contrast", "seed": 2, "image_path": "two.png"},
        ]
        candidate_scores = np.asarray([5.0, 3.0])
        target_mask = np.asarray([True, False])
        natural_ids = np.asarray([10, 11])

        _, raw = compare_with_natural_pool(
            candidates,
            candidate_scores,
            target_mask,
            natural_ids,
            np.asarray([6.0, 4.0]),
        )
        _, matched = compare_with_natural_pool(
            candidates,
            candidate_scores,
            target_mask,
            natural_ids,
            np.asarray([2.0, 1.0]),
        )

        self.assertEqual(raw["best_target_mei_pool_rank"], 2)
        self.assertEqual(matched["best_target_mei_pool_rank"], 1)
        self.assertEqual(raw["target_meis_above_all_natural"], 0)
        self.assertEqual(matched["target_meis_above_all_natural"], 1)


if __name__ == "__main__":
    unittest.main()
