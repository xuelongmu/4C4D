import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import initialize_rgb_points as initializer


class InitializationTests(unittest.TestCase):
    def test_identity_quaternion(self):
        np.testing.assert_allclose(
            initializer.quaternion_to_rotation(np.array([1.0, 0.0, 0.0, 0.0])),
            np.eye(3),
        )

    def test_best_ray_target(self):
        target = np.array([0.2, -0.3, 0.5])
        centers = [np.array([2.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 2.0])]
        cameras = [
            {"center": center, "forward": target - center}
            for center in centers
        ]
        np.testing.assert_allclose(initializer.best_ray_target(cameras), target, atol=1e-12)

    def test_sample_sphere_stays_inside_radius(self):
        points = initializer.sample_sphere(
            np.random.default_rng(4), 1000, np.array([1.0, 2.0, 3.0]), 0.75
        )
        distances = np.linalg.norm(points - np.array([1.0, 2.0, 3.0]), axis=1)
        self.assertLessEqual(float(distances.max()), 0.75)

    def test_triangulated_points_are_radius_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points3D.txt"
            path.write_text(
                "# points\n1 0 0 0 10 20 30 0\n2 5 0 0 40 50 60 0\n",
                encoding="utf-8",
            )
            points, colors = initializer.read_triangulated(path, np.zeros(3), 1.0)
            np.testing.assert_allclose(points, [[0.0, 0.0, 0.0]])
            np.testing.assert_array_equal(colors, [[10, 20, 30]])


if __name__ == "__main__":
    unittest.main()
