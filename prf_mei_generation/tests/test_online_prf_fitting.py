import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tables
import torch

from online_prf_fitting import FixedPrfReadoutFitter


class FixedPrfReadoutFitterTests(unittest.TestCase):
    def test_joint_fit_uses_requested_prf_and_reloads_cache(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_dir = root / "features" / "TEST_MODEL" / "layer2"
            feature_dir.mkdir(parents=True)
            features = rng.normal(size=(30, 5)).astype(np.float32)
            np.save(feature_dir / "features_prf_2.npy", features)

            weights = rng.normal(size=(5, 2))
            betas = features @ weights + 0.01 * rng.normal(size=(30, 2))
            beta_path = root / "betas.hdf5"
            with tables.open_file(beta_path, mode="w") as handle:
                handle.create_array("/", "betas", betas)

            split = {
                "train": list(range(20)),
                "val": list(range(20, 25)),
                "nest": list(range(25, 30)),
            }
            split_path = root / "splits.pkl"
            with open(split_path, "wb") as handle:
                pickle.dump(({"ratio": 1.0}, split, split), handle)

            fitter = FixedPrfReadoutFitter(
                model_name="TEST_MODEL",
                layer_name="layer2",
                split_id=1,
                device=torch.device("cpu"),
                cache_dir=root / "cache",
                feature_root=root / "features",
                split_path=split_path,
                beta_path=beta_path,
            )
            fitted = fitter.fit_targets({0: 2, 1: 2})
            self.assertEqual(fitted[0].target.prf_idx, 2)
            self.assertEqual(fitted[1].target.prf_idx, 2)
            self.assertTrue(fitted[0].cache_path.is_file())
            self.assertGreater(fitted[0].metadata["val_correlation"], 0.99)

            predictions = fitter.predict_natural_images(
                fitted[0],
                split["val"],
            )
            self.assertEqual(predictions.shape, (5,))

            cached = fitter.fit_targets({0: 2, 1: 2})
            torch.testing.assert_close(
                fitted[0].target.weights,
                cached[0].target.weights,
            )


if __name__ == "__main__":
    unittest.main()
