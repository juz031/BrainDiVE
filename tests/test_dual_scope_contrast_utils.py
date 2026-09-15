"""Numerical tests for BrainDiVE dual-scope RMS matching."""

import unittest

import numpy as np
import torch

from dual_scope_contrast_utils import (
    DualScopeContrastConvergenceError,
    global_luma_mean_and_rms,
    match_global_and_prf_rms_contrast_final,
    match_global_and_prf_rms_contrast_torch,
    require_finite_seed_image,
    weighted_luma_mean_and_rms,
)


def gaussian_weight(height, width):
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    return np.exp(-0.5 * (((xx - 0.1) / 0.31) ** 2 + ((yy + 0.2) / 0.48) ** 2))


class DualScopeContrastUtilsTest(unittest.TestCase):
    def test_nonfinite_seed_image_is_retryable(self):
        for invalid_value in (float("nan"), float("inf")):
            images = torch.zeros((1, 3, 8, 8))
            images[0, 0, 0, 0] = invalid_value
            with self.assertRaises(
                DualScopeContrastConvergenceError
            ) as raised:
                require_finite_seed_image(images, stage="test")
            self.assertEqual(raised.exception.reason, "nonfinite_pixels")

    def test_final_solver_hits_reference_tolerance(self):
        generator = torch.Generator().manual_seed(31)
        images = 0.25 + 0.5 * torch.rand((3, 3, 28, 28), generator=generator)
        weight = gaussian_weight(28, 28)
        target = 57.48 / 255.0
        tolerance = 0.01 / 255.0

        result = match_global_and_prf_rms_contrast_final(
            images,
            weight,
            target,
            max_nfev=300,
            tolerance=tolerance,
        )
        _, global_rms = global_luma_mean_and_rms(result.images)
        _, prf_rms = weighted_luma_mean_and_rms(result.images, weight)

        self.assertTrue(bool(torch.all(result.converged)))
        self.assertLessEqual(float((global_rms - target).abs().max()), tolerance)
        self.assertLessEqual(float((prf_rms - target).abs().max()), tolerance)
        self.assertTrue(bool(torch.equal(result.images, result.images.clamp(0, 1))))

    def test_every_step_solver_is_differentiable_and_hits_tolerance(self):
        generator = torch.Generator().manual_seed(43)
        images = (
            0.2 + 0.6 * torch.rand((2, 3, 24, 24), generator=generator)
        ).requires_grad_()
        weight = gaussian_weight(12, 12)
        target = 40.0 / 255.0
        tolerance = 0.01 / 255.0

        result = match_global_and_prf_rms_contrast_torch(
            images,
            weight,
            target,
            iterations=10,
            tolerance=tolerance,
        )
        result.images.square().mean().backward()

        self.assertTrue(bool(torch.all(result.converged)))
        self.assertLessEqual(
            float(
                (result.matched_global_rms - target).abs().max().detach()
            ),
            tolerance,
        )
        self.assertLessEqual(
            float((result.matched_prf_rms - target).abs().max().detach()),
            tolerance,
        )
        self.assertIsNotNone(images.grad)
        self.assertTrue(bool(torch.isfinite(images.grad).all()))
        self.assertGreater(float(images.grad.abs().sum()), 0.0)

    def test_uniform_weight_makes_global_and_prf_statistics_equal(self):
        generator = torch.Generator().manual_seed(59)
        images = 0.2 + 0.6 * torch.rand((2, 3, 20, 20), generator=generator)
        result = match_global_and_prf_rms_contrast_final(
            images,
            np.ones((20, 20), dtype=np.float64),
            0.14,
            tolerance=0.01 / 255.0,
        )

        self.assertTrue(bool(torch.all(result.converged)))
        torch.testing.assert_close(
            result.matched_global_rms,
            result.matched_prf_rms,
            atol=1e-7,
            rtol=0,
        )

    def test_square_root_envelope_is_default(self):
        images = 0.15 + 0.7 * torch.rand((1, 3, 24, 24))
        result = match_global_and_prf_rms_contrast_final(
            images,
            gaussian_weight(24, 24),
            0.17,
        )

        self.assertEqual(result.adjustment_envelope_power, 0.5)


if __name__ == "__main__":
    unittest.main()
