import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow direct unittest discovery without installing this preprocessing module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_depthkit_to_4c4d as converter


class TransformTests(unittest.TestCase):
    def test_identity_scatter_opengl_points_opencv_forward_inward(self):
        pose = {"rotation": [0, 0, 0], "translation": [1, 2, 3]}
        extrinsics = {"rotation": [0, 0, 0], "translation": [0, 0, 0]}
        result = converter.color_camera_to_world(pose, extrinsics)
        np.testing.assert_allclose(result[:3, :3], converter.OPENGL_TO_OPENCV[:3, :3])
        np.testing.assert_allclose(result[:3, 3], [1, 2, 3])

    def test_depth_to_color_extrinsics_are_inverted_for_color_pose(self):
        pose = {"rotation": [0, 0, 0], "translation": [0, 0, 0]}
        extrinsics = {"rotation": [0, 0, 0], "translation": [-0.03, 0, 0]}
        result = converter.color_camera_to_world(
            pose, extrinsics, scatter_basis="opencv"
        )
        np.testing.assert_allclose(result[:3, 3], [0.03, 0, 0], atol=1e-12)

    def test_colmap_quaternion_round_trip(self):
        rotation = cv2.Rodrigues(np.array([0.2, -0.4, 0.1]))[0]
        qw, qx, qy, qz = converter.rotation_matrix_to_colmap_quaternion(rotation)
        reconstructed = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )
        np.testing.assert_allclose(reconstructed, rotation, atol=1e-12)


class DiscoveryTests(unittest.TestCase):
    def test_discovers_single_nested_project(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "person"
            nested.mkdir()
            (nested / "dkproject.json").write_text("{}", encoding="utf-8")
            self.assertEqual(converter.discover_project_root(Path(directory)), nested)

    def test_sensor_number_from_asset(self):
        asset = r"TAKE\Sensor06-SERIAL\S06-SERIAL-color.mp4"
        self.assertEqual(converter.sensor_number_from_asset(asset), 6)

    def test_output_cannot_overwrite_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            with self.assertRaises(converter.ConversionError):
                converter.validate_output_path(project, project)


if __name__ == "__main__":
    unittest.main()
