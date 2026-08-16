import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_rgb_sync as validator


def video(name, timestamps):
    return {"path": name, "stream": {"avg_frame_rate": "30/1"}, "timestamps": timestamps}


class SyncValidationTests(unittest.TestCase):
    def test_aligned_timelines_pass_with_different_end_lengths(self):
        timeline = [index / 30.0 for index in range(12)]
        report = validator.analyze_timelines(
            [video("a.mp4", timeline), video("b.mp4", timeline + [12 / 30.0])],
            [0, 10],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["sample_pts_delta_ms"]["10"], 0.0)
        self.assertTrue(report["warnings"])

    def test_frame_offset_fails_pts_gate(self):
        timeline = [index / 30.0 for index in range(12)]
        shifted = [value + 1 / 30.0 for value in timeline]
        report = validator.analyze_timelines(
            [video("a.mp4", timeline), video("b.mp4", shifted)],
            [0, 10],
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["sample_pts_delta_ms"]["10"], 30.0)

    def test_missing_requested_frame_fails(self):
        report = validator.analyze_timelines(
            [video("a.mp4", [0.0, 1 / 30.0]), video("b.mp4", [0.0])],
            [1],
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("frame 1 is required" in item for item in report["failures"]))

    def test_frame_parser_deduplicates_and_sorts(self):
        self.assertEqual(validator.parse_frame_indices("30,0,30,15"), [0, 15, 30])


if __name__ == "__main__":
    unittest.main()
