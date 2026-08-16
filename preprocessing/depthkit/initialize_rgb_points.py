#!/usr/bin/env python3
"""Create a depth-free 4C4D seed cloud from calibrated RGB cameras.

Reliable COLMAP-triangulated RGB points can be retained, then the remaining
budget is filled inside volumes derived only from the calibrated camera rig.
Colors are the median of visible RGB projections at one synchronized frame.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def read_cameras(path: Path) -> dict[int, np.ndarray]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        if model != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera, got {model}")
        fx, fy, cx, cy = map(float, fields[4:8])
        result[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return result


def read_images(path: Path, frame: int):
    """Per-camera poses for `frame`, named after that frame's extracted image.

    convert_depthkit_to_4c4d.py emits exactly one pose entry per camera, always
    named camXX_0000.png, because the rig is fixed for the whole clip. So for
    any nonzero --frame there is no matching entry; reuse the frame-zero poses
    and point their names at the requested frame instead of failing.
    """
    entries = [
        fields for fields in
        (line.split() for line in path.read_text(encoding="utf-8").splitlines())
        if len(fields) >= 10
    ]
    suffix = f"_{frame:04d}.png"
    matching = [fields for fields in entries if fields[9].endswith(suffix)]
    if not matching:
        matching = [fields for fields in entries if fields[9].endswith("_0000.png")]

    cameras = []
    for fields in matching:
        rotation = quaternion_to_rotation(np.asarray(fields[1:5], dtype=float))
        translation = np.asarray(fields[5:8], dtype=float)
        cameras.append({
            "rotation": rotation,
            "translation": translation,
            "camera_id": int(fields[8]),
            "name": re.sub(r"_\d{4}\.png$", suffix, fields[9]),
            "center": -rotation.T @ translation,
            "forward": rotation.T @ np.array([0.0, 0.0, 1.0]),
        })
    if not cameras:
        raise ValueError(f"No calibrated images found for frame {frame}")
    return cameras


def best_ray_target(cameras) -> np.ndarray:
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for camera in cameras:
        d = camera["forward"] / np.linalg.norm(camera["forward"])
        projector = np.eye(3) - np.outer(d, d)
        a += projector
        b += projector @ camera["center"]
    return np.linalg.solve(a, b)


def sample_sphere(rng, count: int, center: np.ndarray, radius: float) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = radius * rng.random(count)[:, None] ** (1.0 / 3.0)
    return center + directions * distances


def read_triangulated(path: Path | None, center: np.ndarray, max_radius: float):
    points, colors = [], []
    if path is None or not path.exists():
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        point = np.asarray(fields[1:4], dtype=float)
        if np.all(np.isfinite(point)) and np.linalg.norm(point - center) <= max_radius:
            points.append(point)
            colors.append(np.asarray(fields[4:7], dtype=np.uint8))
    return np.asarray(points).reshape(-1, 3), np.asarray(colors, dtype=np.uint8).reshape(-1, 3)


def project_colors(points, cameras, intrinsics, image_dir: Path):
    sampled = []
    for camera in cameras:
        image = cv2.imread(str(image_dir / camera["name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_dir / camera["name"])
        local = (camera["rotation"] @ points.T).T + camera["translation"]
        projected = (intrinsics[camera["camera_id"]] @ local.T).T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        valid = (
            (local[:, 2] > 0) & (u >= 0) & (u < image.shape[1]) &
            (v >= 0) & (v < image.shape[0])
        )
        colors = np.full((len(points), 3), np.nan, dtype=np.float32)
        colors[valid] = image[v[valid], u[valid], ::-1]
        sampled.append(colors)
    stacked = np.stack(sampled, axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        colors = np.nanmedian(stacked, axis=0)
    visible = np.sum(np.isfinite(stacked[:, :, 0]), axis=0)
    colors[~np.isfinite(colors)] = 127
    return np.clip(colors, 0, 255).astype(np.uint8), visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--triangulated-points", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=75000)
    parser.add_argument("--near-ratio", type=float, default=0.8)
    parser.add_argument("--near-radius-rigs", type=float, default=0.8)
    parser.add_argument("--far-radius-rigs", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()

    sparse = args.scene / "sparse" / "0"
    intrinsics = read_cameras(sparse / "cameras.txt")
    cameras = read_images(sparse / "images.txt", args.frame)
    target = best_ray_target(cameras)
    rig_radius = float(np.median([np.linalg.norm(c["center"] - target) for c in cameras]))
    far_radius = args.far_radius_rigs * rig_radius
    triangulated, triangulated_colors = read_triangulated(
        args.triangulated_points, target, far_radius
    )
    if len(triangulated) > args.num_points:
        triangulated = triangulated[:args.num_points]
        triangulated_colors = triangulated_colors[:args.num_points]

    remaining = args.num_points - len(triangulated)
    near_count = int(round(remaining * args.near_ratio))
    rng = np.random.default_rng(args.seed)
    generated = np.concatenate([
        sample_sphere(rng, near_count, target, args.near_radius_rigs * rig_radius),
        sample_sphere(rng, remaining - near_count, target, far_radius),
    ])
    generated_colors, visibility = project_colors(
        generated, cameras, intrinsics, args.scene / "images"
    )
    points = np.concatenate([triangulated, generated])
    colors = np.concatenate([triangulated_colors, generated_colors])

    scene_points = sparse / "points3D.txt"
    output = args.output or scene_points
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# RGB-only calibrated-volume initialization; no depth data used\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for point_id, (point, color) in enumerate(zip(points, colors), 1):
            handle.write(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )

    report = {
        "method": "rgb_triangulation_plus_calibrated_rig_volume",
        "depthDataUsed": False,
        "frame": args.frame,
        "pointCount": len(points),
        "triangulatedPointCount": len(triangulated),
        "generatedPointCount": len(generated),
        "rigTarget": target.tolist(),
        "rigRadius": rig_radius,
        "nearRadius": args.near_radius_rigs * rig_radius,
        "farRadius": far_radius,
        "generatedVisibleInAtLeastTwoCameras": int(np.sum(visibility >= 2)),
        "generatedVisibleInNoCameras": int(np.sum(visibility == 0)),
        "outputPath": str(output),
    }
    report_path = output.with_suffix(".rgb_init.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Only claim the scene uses this cloud when it actually replaced the scene's
    # active points3D.txt. For a detached --output the manifest would otherwise
    # describe a cloud the dataset never loads, misreporting provenance.
    manifest_path = args.scene / "conversion_manifest.json"
    replaced_scene_points = output.resolve() == scene_points.resolve()
    if manifest_path.exists() and replaced_scene_points:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["initialPointCount"] = len(points)
        manifest["initialPointSource"] = report["method"]
        manifest["depthDataUsed"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif manifest_path.exists():
        print(
            f"# wrote {output} without touching {manifest_path.name}: "
            "the scene's active point cloud is unchanged",
            file=sys.stderr,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
