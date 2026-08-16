import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_rgb_calibration as validator


class CalibrationValidationTests(unittest.TestCase):
    def test_camera_key_removes_only_numeric_frame_suffix(self):
        self.assertEqual(validator.camera_key("nested/cam03_0086.png"), "cam03")
        self.assertEqual(validator.camera_key("left_camera.png"), "left_camera")

    def test_summary_includes_tail_metrics(self):
        summary = validator.summarize(np.arange(100, dtype=float))
        self.assertEqual(summary["count"], 100)
        self.assertAlmostEqual(summary["median_px"], 49.5)
        self.assertGreater(summary["p95_px"], summary["p90_px"])

    def test_connected_components_include_isolated_cameras(self):
        components = validator.connected_components(
            {"cam00", "cam01", "cam02"}, {("cam00", "cam01")}
        )
        self.assertEqual(components, [["cam00", "cam01"], ["cam02"]])

    def test_quality_gate_rejects_good_median_with_bad_tail(self):
        pair_reports = [
            {
                "pair": ["cam00_0000.png", "cam01_0000.png"],
                "cameras": ["cam00", "cam01"],
                "count": 100,
                "median_px": 0.5,
                "p90_px": 40.0,
                "colmap_pairwise_control": {"p90_px": 0.8},
            }
        ]
        thresholds = {
            **validator.DEFAULT_THRESHOLDS,
            "min_reliable_neighbors": 1,
        }
        gate = validator.quality_gate(
            {"cam00", "cam01"},
            pair_reports,
            {"median_px": 0.5, "p90_px": 40.0},
            {"p90_px": 0.8},
            thresholds,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(pair_reports[0]["pose_valid"])
        self.assertIn(pair_reports[0]["pair"], gate["failing_pairs"])

    def test_quality_gate_requires_connected_reliable_graph(self):
        pair_reports = []
        for camera_a, camera_b in (("cam00", "cam01"), ("cam02", "cam03")):
            pair_reports.append({
                "pair": [f"{camera_a}_0000.png", f"{camera_b}_0000.png"],
                "cameras": [camera_a, camera_b],
                "count": 50,
                "median_px": 0.5,
                "p90_px": 1.0,
                "colmap_pairwise_control": {"p90_px": 0.8},
            })
        thresholds = {
            **validator.DEFAULT_THRESHOLDS,
            "min_reliable_neighbors": 1,
        }
        gate = validator.quality_gate(
            {"cam00", "cam01", "cam02", "cam03"},
            pair_reports,
            {"median_px": 0.5, "p90_px": 1.0},
            {"p90_px": 0.8},
            thresholds,
        )
        self.assertFalse(gate["passed"])
        self.assertIn(
            "calibration-consistent camera graph is disconnected", gate["failures"]
        )

    def test_two_camera_rig_caps_neighbor_requirement(self):
        pair_reports = [{
            "pair": ["cam00_0000.png", "cam01_0000.png"],
            "cameras": ["cam00", "cam01"],
            "count": 50,
            "median_px": 0.5,
            "p90_px": 1.0,
            "colmap_pairwise_control": {"p90_px": 0.8},
        }]
        gate = validator.quality_gate(
            {"cam00", "cam01"},
            pair_reports,
            {"median_px": 0.5, "p90_px": 1.0},
            {"p90_px": 0.8},
            validator.DEFAULT_THRESHOLDS.copy(),
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["effective_min_reliable_neighbors"], 1)


if __name__ == "__main__":
    unittest.main()
