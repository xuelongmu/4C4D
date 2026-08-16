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
import re
import sqlite3
from pathlib import Path

import numpy as np


MAX_IMAGE_ID = 2**31 - 1
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
TIMESTAMP_SUFFIX = re.compile(r"_\d+$")

DEFAULT_THRESHOLDS = {
    "min_pair_matches": 30,
    "max_control_p90_px": 4.0,
    "max_pose_median_px": 2.0,
    "max_pose_p90_px": 4.0,
    "min_reliable_neighbors": 2,
}


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
        "p95_px": float(np.percentile(values, 95)),
        "under_1px": float(np.mean(values < 1)),
        "under_2px": float(np.mean(values < 2)),
        "under_4px": float(np.mean(values < 4)),
    }


def camera_key(image_name: str) -> str:
    """Return a stable camera label for names such as cam03_0086.png."""
    return TIMESTAMP_SUFFIX.sub("", Path(image_name).stem)


def connected_components(nodes: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for node_a, node_b in edges:
        if node_a == node_b:
            continue
        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

    components = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        stack, component = [seed], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        remaining -= component
        components.append(sorted(component))
    return sorted(components, key=lambda value: (-len(value), value))


def quality_gate(
    camera_names: set[str],
    pair_reports: list[dict],
    supplied_pose: dict,
    control: dict | None,
    thresholds: dict,
) -> dict:
    reliable_edges = set()
    verified_edges = set()
    failing_pairs = []
    neighbor_sets = {camera: set() for camera in camera_names}

    for pair in pair_reports:
        camera_a, camera_b = pair["cameras"]
        if camera_a == camera_b:
            continue
        edge = tuple(sorted((camera_a, camera_b)))
        control_summary = pair.get("colmap_pairwise_control")
        control_ok = bool(
            pair["count"] >= thresholds["min_pair_matches"]
            and control_summary is not None
            and control_summary["p90_px"] <= thresholds["max_control_p90_px"]
        )
        pose_ok = bool(
            pair["median_px"] <= thresholds["max_pose_median_px"]
            and pair["p90_px"] <= thresholds["max_pose_p90_px"]
        )
        pair["control_valid"] = control_ok
        pair["pose_valid"] = pose_ok
        pair["reliable"] = control_ok and pose_ok
        if control_ok:
            verified_edges.add(edge)
        if pair["reliable"]:
            reliable_edges.add(edge)
            neighbor_sets[camera_a].add(camera_b)
            neighbor_sets[camera_b].add(camera_a)
        elif control_ok:
            failing_pairs.append(pair["pair"])

    verified_components = connected_components(camera_names, verified_edges)
    reliable_components = connected_components(camera_names, reliable_edges)
    required_neighbors = min(
        thresholds["min_reliable_neighbors"], max(len(camera_names) - 1, 0)
    )
    weak_cameras = sorted(
        camera for camera, neighbors in neighbor_sets.items()
        if len(neighbors) < required_neighbors
    )
    failures = []
    if control is None:
        failures.append("no COLMAP pairwise control matrices were available")
    elif control["p90_px"] > thresholds["max_control_p90_px"]:
        failures.append(
            "pairwise control p90 exceeds "
            f"{thresholds['max_control_p90_px']:.3g}px"
        )
    if supplied_pose["median_px"] > thresholds["max_pose_median_px"]:
        failures.append(
            f"supplied-pose median exceeds {thresholds['max_pose_median_px']:.3g}px"
        )
    if supplied_pose["p90_px"] > thresholds["max_pose_p90_px"]:
        failures.append(
            f"supplied-pose p90 exceeds {thresholds['max_pose_p90_px']:.3g}px"
        )
    if len(verified_components) > 1:
        failures.append("verified RGB match graph is disconnected")
    if len(reliable_components) > 1:
        failures.append("calibration-consistent camera graph is disconnected")
    if weak_cameras:
        failures.append(
            "cameras lack the minimum number of reliable neighbors: "
            + ", ".join(weak_cameras)
        )

    return {
        "passed": not failures,
        "thresholds": thresholds,
        "effective_min_reliable_neighbors": required_neighbors,
        "failures": failures,
        "failing_pairs": failing_pairs,
        "weak_cameras": weak_cameras,
        "verified_graph_components": verified_components,
        "reliable_graph_components": reliable_components,
        "reliable_neighbors": {
            camera: sorted(neighbors) for camera, neighbors in sorted(neighbor_sets.items())
        },
    }


def analyze(model: Path, database: Path, *, thresholds: dict | None = None) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    cameras = read_text_model(model)
    cameras_by_key = {camera_key(name): camera for name, camera in cameras.items()}
    names, keypoints, pairs = load_database(database)
    supplied_errors, control_errors, pair_reports = [], [], []
    per_camera_errors: dict[str, list[np.ndarray]] = {
        name: [] for name in cameras_by_key
    }
    per_camera_matches = {name: 0 for name in cameras_by_key}
    for image_id_a, image_id_b, matches, control_f in pairs:
        name_a, name_b = names[image_id_a], names[image_id_b]
        camera_name_a, camera_name_b = camera_key(name_a), camera_key(name_b)
        camera_a = cameras.get(name_a, cameras_by_key.get(camera_name_a))
        camera_b = cameras.get(name_b, cameras_by_key.get(camera_name_b))
        if camera_a is None or camera_b is None:
            continue
        points_a = keypoints[image_id_a][matches[:, 0]]
        points_b = keypoints[image_id_b][matches[:, 1]]
        errors = sampson_error(
            fundamental(camera_a, camera_b), points_a, points_b
        )
        supplied_errors.append(errors)
        per_camera_errors[camera_name_a].append(errors)
        per_camera_errors[camera_name_b].append(errors)
        per_camera_matches[camera_name_a] += len(errors)
        per_camera_matches[camera_name_b] += len(errors)
        pair_report = {
            "pair": [name_a, name_b],
            "cameras": [camera_name_a, camera_name_b],
            **summarize(errors),
        }
        if control_f is not None:
            control = sampson_error(control_f, points_a, points_b)
            control_errors.append(control)
            pair_report["colmap_f_median_px"] = float(np.median(control))
            pair_report["colmap_pairwise_control"] = summarize(control)
        pair_reports.append(pair_report)

    if not supplied_errors:
        raise ValueError("No database image pairs matched names in the supplied model")
    supplied_summary = summarize(np.concatenate(supplied_errors))
    control_summary = summarize(np.concatenate(control_errors)) if control_errors else None
    gate = quality_gate(
        set(cameras_by_key), pair_reports, supplied_summary, control_summary, thresholds
    )
    camera_reports = []
    for name in sorted(cameras_by_key):
        errors = per_camera_errors[name]
        camera_report = {
            "camera": name,
            "verified_match_count": per_camera_matches[name],
            "reliable_neighbors": gate["reliable_neighbors"][name],
            "reliable_neighbor_count": len(gate["reliable_neighbors"][name]),
        }
        if errors:
            camera_report["supplied_pose"] = summarize(np.concatenate(errors))
        camera_reports.append(camera_report)

    report = {
        "model": str(model),
        "database": str(database),
        "pair_count": len(pair_reports),
        "supplied_pose": supplied_summary,
        "quality_gate": gate,
        "cameras": camera_reports,
        "pairs": sorted(pair_reports, key=lambda value: value["median_px"]),
    }
    if control_summary:
        report["colmap_pairwise_control"] = control_summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pair-matches", type=int, default=30)
    parser.add_argument("--max-control-p90-px", type=float, default=4.0)
    parser.add_argument("--max-pose-median-px", type=float, default=2.0)
    parser.add_argument("--max-pose-p90-px", type=float, default=4.0)
    parser.add_argument("--min-reliable-neighbors", type=int, default=2)
    parser.add_argument(
        "--fail-on-quality-gate", action="store_true",
        help="Exit with status 2 when the calibration quality gate fails",
    )
    args = parser.parse_args()
    thresholds = {
        "min_pair_matches": args.min_pair_matches,
        "max_control_p90_px": args.max_control_p90_px,
        "max_pose_median_px": args.max_pose_median_px,
        "max_pose_p90_px": args.max_pose_p90_px,
        "min_reliable_neighbors": args.min_reliable_neighbors,
    }
    report = analyze(args.model, args.database, thresholds=thresholds)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_quality_gate and not report["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
