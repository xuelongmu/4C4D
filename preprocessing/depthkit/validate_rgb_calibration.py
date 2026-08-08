#!/usr/bin/env python3
"""Measure a COLMAP text model against verified RGB feature matches.

The supplied camera model is evaluated with Sampson epipolar error. COLMAP's
independently estimated per-pair fundamental matrices are reported as a control:
if those are accurate but the supplied-pose errors are large, feature matching
worked and the imported calibration convention is the likely failure.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


MAX_IMAGE_ID = 2**31 - 1
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def read_text_model(model: Path) -> dict[str, dict[str, np.ndarray]]:
    intrinsics = {}
    for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or line.startswith("#"):
            continue
        if fields[1] != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera, got {fields[1]}")
        intrinsics[int(fields[0])] = np.array([
            [float(fields[4]), 0.0, float(fields[6])],
            [0.0, float(fields[5]), float(fields[7])],
            [0.0, 0.0, 1.0],
        ])

    images = {}
    for line in (model / "images.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) < 10
            or line.startswith("#")
            or not fields[9].lower().endswith(IMAGE_EXTENSIONS)
        ):
            continue
        camera_id = int(fields[8])
        images[fields[9]] = {
            "rotation": quaternion_to_rotation(np.asarray(fields[1:5], dtype=float)),
            "translation": np.asarray(fields[5:8], dtype=float),
            "intrinsics": intrinsics[camera_id],
        }
    return images


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental(camera_a, camera_b) -> np.ndarray:
    relative_rotation = camera_b["rotation"] @ camera_a["rotation"].T
    relative_translation = (
        camera_b["translation"]
        - relative_rotation @ camera_a["translation"]
    )
    essential = skew(relative_translation) @ relative_rotation
    return (
        np.linalg.inv(camera_b["intrinsics"]).T
        @ essential
        @ np.linalg.inv(camera_a["intrinsics"])
    )


def sampson_error(matrix: np.ndarray, points_a, points_b) -> np.ndarray:
    homogeneous_a = np.column_stack([points_a, np.ones(len(points_a))])
    homogeneous_b = np.column_stack([points_b, np.ones(len(points_b))])
    lines_b = (matrix @ homogeneous_a.T).T
    lines_a = (matrix.T @ homogeneous_b.T).T
    numerator = np.sum(homogeneous_b * lines_b, axis=1) ** 2
    denominator = (
        lines_b[:, 0]**2 + lines_b[:, 1]**2
        + lines_a[:, 0]**2 + lines_a[:, 1]**2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-15))


def load_database(database: Path):
    connection = sqlite3.connect(database)
    names = dict(connection.execute("SELECT image_id, name FROM images"))
    keypoints = {
        image_id: np.frombuffer(data, dtype=np.float32).reshape(rows, cols)[:, :2]
        for image_id, rows, cols, data in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        )
    }
    pairs = []
    for pair_id, rows, cols, data, stored_f in connection.execute(
        "SELECT pair_id, rows, cols, data, F FROM two_view_geometries WHERE rows > 0"
    ):
        image_id_b = pair_id % MAX_IMAGE_ID
        image_id_a = (pair_id - image_id_b) // MAX_IMAGE_ID
        matches = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)
        control_f = (
            np.frombuffer(stored_f, dtype=np.float64).reshape(3, 3)
            if stored_f else None
        )
        pairs.append((image_id_a, image_id_b, matches, control_f))
    connection.close()
    return names, keypoints, pairs


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "under_1px": float(np.mean(values < 1)),
        "under_2px": float(np.mean(values < 2)),
        "under_4px": float(np.mean(values < 4)),
    }


def analyze(model: Path, database: Path) -> dict:
    cameras = read_text_model(model)
    names, keypoints, pairs = load_database(database)
    supplied_errors, control_errors, pair_reports = [], [], []
    for image_id_a, image_id_b, matches, control_f in pairs:
        name_a, name_b = names[image_id_a], names[image_id_b]
        if name_a not in cameras or name_b not in cameras:
            continue
        points_a = keypoints[image_id_a][matches[:, 0]]
        points_b = keypoints[image_id_b][matches[:, 1]]
        errors = sampson_error(
            fundamental(cameras[name_a], cameras[name_b]), points_a, points_b
        )
        supplied_errors.append(errors)
        pair_report = {"pair": [name_a, name_b], **summarize(errors)}
        if control_f is not None:
            control = sampson_error(control_f, points_a, points_b)
            control_errors.append(control)
            pair_report["colmap_f_median_px"] = float(np.median(control))
        pair_reports.append(pair_report)

    if not supplied_errors:
        raise ValueError("No database image pairs matched names in the supplied model")
    report = {
        "model": str(model),
        "database": str(database),
        "pair_count": len(pair_reports),
        "supplied_pose": summarize(np.concatenate(supplied_errors)),
        "pairs": sorted(pair_reports, key=lambda value: value["median_px"]),
    }
    if control_errors:
        report["colmap_pairwise_control"] = summarize(np.concatenate(control_errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.model, args.database)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
