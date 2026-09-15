"""CPU-only tests with synthetic subject-specific stimulus files."""

import contextlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import calculate_nsd_contrast_statistics as stats


class SubjectSelectionTests(unittest.TestCase):
    def test_default_paths_preserve_s1(self):
        args = stats.parse_args([])
        self.assertEqual(args.subject, 1)
        self.assertEqual(stats.DEFAULT_SPLIT_ROOT,
                         Path("/user_data/junruz/prf_models/concat/split_1_zscore"))
        self.assertEqual(args.stimulus_file, stats.DEFAULT_STIMULUS_DIR / "S1_stimuli_224.h5py")
        self.assertEqual(args.split_file, stats.DEFAULT_SPLIT_ROOT / "S1/data_splits_S1.pkl")
        self.assertEqual(args.output_json, stats.DEFAULT_SPLIT_ROOT / "S1/nsd_contrast_statistics.json")
        self.assertIsNone(args.output_per_image)

    def test_subject_changes_all_defaults(self):
        for subject in range(1, 9):
            with self.subTest(subject=subject):
                args = stats.parse_args(["--subject", str(subject)])
                self.assertEqual(args.stimulus_file.name, f"S{subject}_stimuli_224.h5py")
                self.assertEqual(args.split_file.name, f"data_splits_S{subject}.pkl")
                self.assertEqual(args.split_file.parent.name, f"S{subject}")
                self.assertEqual(args.output_json.parent.name, f"S{subject}")

    def test_custom_roots(self):
        args = stats.parse_args([
            "--subject", "2", "--stimulus-dir", "/custom/stimuli",
            "--split-root", "/custom/splits",
        ])
        self.assertEqual(args.stimulus_file, Path("/custom/stimuli/S2_stimuli_224.h5py"))
        self.assertEqual(args.split_file, Path("/custom/splits/S2/data_splits_S2.pkl"))
        self.assertEqual(args.output_json, Path("/custom/splits/S2/nsd_contrast_statistics.json"))

    def test_explicit_paths_override_defaults_independently(self):
        for option, attribute in [
            ("--stimulus-file", "stimulus_file"),
            ("--split-file", "split_file"),
            ("--output-json", "output_json"),
            ("--output-per-image", "output_per_image"),
        ]:
            with self.subTest(option=option):
                defaults = stats.parse_args(["--subject", "3"])
                args = stats.parse_args(["--subject", "3", option, "/custom/file"])
                self.assertEqual(getattr(args, attribute), Path("/custom/file"))
                for other in ("stimulus_file", "split_file", "output_json", "output_per_image"):
                    if other != attribute:
                        self.assertEqual(getattr(args, other), getattr(defaults, other))

    def test_invalid_subjects(self):
        for subject in ("0", "9", "S1", "1.5"):
            with self.subTest(subject=subject), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    stats.parse_args(["--subject", subject])

    def test_two_subjects_save_separate_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject, brightness in ((1, 40), (2, 180)):
                subject_dir = root / f"S{subject}"
                subject_dir.mkdir()
                images = np.full((4, 3, 4, 4), brightness, dtype=np.uint8)
                images[3] = 255  # Validation image must not enter default statistics.
                with h5py.File(root / f"S{subject}_stimuli_224.h5py", "w") as handle:
                    handle.create_dataset("stimuli", data=images)
                with (subject_dir / f"data_splits_S{subject}.pkl").open("wb") as handle:
                    pickle.dump({1: {"train": [0, 1], "nest": [1, 2], "val": [3]}}, handle)
                argv = [
                    "stats", "--subject", str(subject), "--stimulus-dir", str(root),
                    "--split-root", str(root), "--batch-size", "2",
                    "--output-per-image", str(subject_dir / "per_image.npz"),
                ]
                with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    stats.main()

            # Read both after both runs: subject 2 must not overwrite subject 1.
            for subject, brightness in ((1, 40), (2, 180)):
                subject_dir = root / f"S{subject}"
                summary = json.loads((subject_dir / "nsd_contrast_statistics.json").read_text())
                self.assertEqual(summary["subject"], subject)
                self.assertEqual(summary["image_count"], 3)
                self.assertAlmostEqual(summary["luminance_mean"]["byte_0_255"]["mean"], brightness, places=4)
                self.assertAlmostEqual(summary["rms_contrast"]["byte_0_255"]["mean"], 0)
                with np.load(subject_dir / "per_image.npz") as data:
                    np.testing.assert_array_equal(data["image_indices"], [0, 1, 2])
                    np.testing.assert_allclose(data["luminance_mean"], brightness / 255, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
