from pathlib import Path
import unittest

import torch

from maximize_prf_activation import (
    apply_task_config,
    build_arg_parser,
    build_optimization_config,
    finalize_args,
)
from max_activation_utils import (
    OptimizationConfig,
    RMSContrastConstraint,
    apply_rms_contrast_constraint,
    batch_luminance_mean,
    batch_rms_contrast,
    image_luminance,
    initialize_image_param,
    optimize_image,
    optimize_image_batch,
    translate_image,
)


class MeanObjective(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(1, 2, 3), keepdim=True)


class ContrastConstraintTests(unittest.TestCase):
    def test_rms_contrast_is_computed_after_luminance_conversion(self):
        image = torch.tensor(
            [
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                    [[0.0, 0.0]],
                ]
            ]
        )
        luminance = image_luminance(image)
        expected = torch.tensor([[[[0.2126, 0.7152]]]])
        torch.testing.assert_close(luminance, expected)
        torch.testing.assert_close(
            batch_luminance_mean(image),
            expected.mean().reshape(1),
        )
        torch.testing.assert_close(
            batch_rms_contrast(image),
            expected.std(unbiased=False).reshape(1),
        )

    def test_mean_jitter_does_not_wrap_opposite_edge(self):
        image = torch.zeros((1, 1, 4, 4))
        image[..., -1, -1] = 1.0
        translated = translate_image(image, shift_y=1, shift_x=1, mode="mean")
        legacy = translate_image(image, shift_y=1, shift_x=1, mode="roll")

        self.assertEqual(float(legacy[..., 0, 0]), 1.0)
        self.assertNotEqual(float(translated[..., 0, 0]), 1.0)
        self.assertEqual(float(translated[..., 0, 0]), float(image.mean()))

    def test_json_task_config_converts_byte_contrast_to_unit_space(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "clip_rn50_v1_contrast.example.json"
        )
        args = build_arg_parser().parse_args(
            ["--task_config", str(config_path), "--task_id", "1"]
        )
        args = finalize_args(apply_task_config(args))
        config = build_optimization_config(args)

        self.assertAlmostEqual(config.train_contrast.target, 16.0 / 255.0)
        self.assertAlmostEqual(config.post_step_contrast.target, 16.0 / 255.0)
        self.assertAlmostEqual(config.final_contrast.target, 16.0 / 255.0)
        self.assertAlmostEqual(config.final_contrast.target_mean, 111.0 / 255.0)

    def test_max_projection_preserves_mean_and_caps_contrast(self):
        image = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        constrained = apply_rms_contrast_constraint(
            image,
            RMSContrastConstraint(target=0.1, mode="max"),
        )

        torch.testing.assert_close(constrained.mean(), image.mean())
        torch.testing.assert_close(
            batch_rms_contrast(constrained),
            torch.tensor([0.1]),
        )

    def test_max_projection_does_not_expand_low_contrast_image(self):
        image = torch.tensor([[[[0.45, 0.55], [0.45, 0.55]]]])
        constrained = apply_rms_contrast_constraint(
            image,
            RMSContrastConstraint(target=0.2, mode="max"),
        )
        torch.testing.assert_close(constrained, image)

    def test_match_projection_sets_mean_and_contrast(self):
        image = torch.tensor([[[[0.3, 0.7], [0.3, 0.7]]]])
        constrained = apply_rms_contrast_constraint(
            image,
            RMSContrastConstraint(target=0.08, mode="match", target_mean=0.4),
        )

        torch.testing.assert_close(constrained.mean(), torch.tensor(0.4))
        torch.testing.assert_close(
            batch_rms_contrast(constrained),
            torch.tensor([0.08]),
        )

    def test_match_projection_compensates_for_clipping(self):
        image = torch.linspace(0.0, 1.0, 100).reshape(1, 1, 10, 10)
        constrained = apply_rms_contrast_constraint(
            image,
            RMSContrastConstraint(
                target=0.15,
                mode="match",
                target_mean=0.1,
            ),
        )

        torch.testing.assert_close(
            constrained.mean(),
            torch.tensor(0.1),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            batch_rms_contrast(constrained),
            torch.tensor([0.15]),
            atol=2e-5,
            rtol=0,
        )

    def test_optimizer_applies_post_step_and_final_constraints(self):
        torch.manual_seed(0)
        image_param = initialize_image_param(
            init_mode="random",
            image_size=8,
            device=torch.device("cpu"),
        )
        config = OptimizationConfig(
            steps=3,
            lr=0.05,
            jitter=0,
            record_every=1,
            tv_weight=0.0,
            l2_weight=0.0,
            post_step_contrast=RMSContrastConstraint(target=0.05, mode="max"),
            final_contrast=RMSContrastConstraint(
                target=0.04,
                mode="match",
                target_mean=0.4,
            ),
        )

        result = optimize_image(MeanObjective(), image_param, config)

        torch.testing.assert_close(
            batch_luminance_mean(result.image).mean(),
            torch.tensor(0.4),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            batch_rms_contrast(result.image),
            torch.tensor([0.04]),
            atol=2e-5,
            rtol=0,
        )
        self.assertIn("image_rms_contrast", result.history[-1])
        self.assertIn("model_rms_contrast", result.history[-1])

    def test_batch_optimizer_returns_one_constrained_result_per_seed(self):
        torch.manual_seed(0)
        images = torch.rand((3, 3, 8, 8)) * 0.6 + 0.2
        image_param = torch.logit(images).requires_grad_(True)
        config = OptimizationConfig(
            steps=3,
            lr=0.05,
            jitter=0,
            record_every=1,
            tv_weight=0.0,
            l2_weight=0.0,
            post_step_contrast=RMSContrastConstraint(target=0.05, mode="max"),
            final_contrast=RMSContrastConstraint(
                target=0.04,
                mode="match",
                target_mean=0.4,
            ),
        )

        result = optimize_image_batch(MeanObjective(), image_param, config)

        self.assertEqual(tuple(result.images.shape), (3, 3, 8, 8))
        self.assertEqual(tuple(result.best_scores.shape), (3,))
        torch.testing.assert_close(
            batch_luminance_mean(result.images),
            torch.full((3,), 0.4),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            batch_rms_contrast(result.images),
            torch.full((3,), 0.04),
            atol=2e-5,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
