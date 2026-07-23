from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.fit_v0 import fit_dataset, index_dataset, write_results


DATASET_ROOT = Path("images/real/calibration_B")


class FitV0Tests(unittest.TestCase):
    def test_indexes_expected_bursts(self) -> None:
        groups = index_dataset(DATASET_ROOT)

        self.assertEqual(
            set(groups),
            {"00", "01", "05", "18", "19", "19_2", "22", "23", "24", "28", "30", "30_2"},
        )
        self.assertTrue(all(len(files) == 10 for files in groups.values()))

    def test_fits_capture_domains_and_effective_degradation(self) -> None:
        result = fit_dataset(DATASET_ROOT)

        self.assertEqual(result["schema_version"], "v0")
        self.assertEqual(result["groups"]["19_2"]["focal_35mm_median"], 69.0)
        self.assertEqual(result["groups"]["30_2"]["focal_35mm_median"], 69.0)
        self.assertEqual(result["groups"]["23"]["focal_35mm_median"], 47.0)
        self.assertIsNone(result["groups"]["01"]["auto_exposure"])
        self.assertEqual(result["groups"]["01"]["exposure_mode_source"], "missing_in_timeburst_exif")
        self.assertNotIn("30", result["domains"]["47mm"]["groups"])
        self.assertIn("30", result["domains"]["69mm"]["groups"])

        point_grid = result["domains"]["47mm"]["point_grid_7px"]
        self.assertGreater(point_grid["detected_dot_samples"], 300)
        self.assertTrue(all(0.35 < value < 0.60 for value in point_grid["scale_camera_per_source"]))
        self.assertTrue(all(5.0 < value < 13.0 for value in point_grid["observed_fwhm_camera_px"]))

        checker = result["domains"]["69mm"]["checkerboard"]
        self.assertTrue(all(0.40 < value < 0.60 for value in checker["scale_camera_per_source"]))
        self.assertTrue(3.0 < checker["edge_gradient_fwhm_camera_px"] < 10.0)

        self.assertTrue(result["photometry"]["auto_exposure_detected"])
        self.assertGreater(result["photometry"]["white_to_gray128_rgb_ratio"][1], 1.2)
        self.assertIn("21_asymmetric_grid", result["recommended_v2"]["required_patterns"])

    def test_writes_json_and_markdown_results(self) -> None:
        result = fit_dataset(DATASET_ROOT)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_results(result, output)

            self.assertTrue((output / "v0_parameters.json").is_file())
            report = (output / "v0_report.md").read_text(encoding="utf-8")
            self.assertIn("第二版拍摄", report)
            self.assertIn("47 mm", report)
            self.assertIn("69 mm", report)


if __name__ == "__main__":
    unittest.main()
