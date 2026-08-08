#!/usr/bin/env python3
"""Convert a Depthkit/Scatter multi-camera take into a 4C4D COLMAP dataset.

The converter reads camera calibration and recording metadata from dkproject.json,
extracts synchronized RGB frames, undistorts them to a PINHOLE model, writes a
fixed-camera COLMAP text model, and can initialize points3D.txt from frame-zero
depth maps.

Depthkit/Scatter convention assumptions used by default:
  * worldExtrinsics.world is a camera-to-world pose.
  * Its saved rig transform is left-handed. A Z reflection is applied on both
    the world and camera sides to produce a proper right-handed rotation.
  * Color-source extrinsics transform depth-camera coordinates to color-camera
    coordinates, so their inverse places the RGB optical center in the rig.

These assumptions are written to conversion_manifest.json. Validate the result
with --validate-only first and inspect the reported look-at score before training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in an unprepared env
    raise SystemExit(
        "This converter requires numpy and opencv-python. Install them with: "
        "python -m pip install numpy opencv-python"
    ) from exc


SCATTER_HANDEDNESS = np.diag([1.0, 1.0, -1.0, 1.0])
# Retained for callers which explicitly request the legacy OpenGL assumption.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_PATTERN_TOKENS = ("%10T", "%5F")
SENSOR_NUMBER_RE = re.compile(r"Sensor(\d+)-", re.IGNORECASE)


class ConversionError(RuntimeError):
    """A user-actionable input or conversion error."""


@dataclass
class CameraInput:
    sensor_number: int
    camera_index: int
    device_id: str
    color_path: Path
    depth_paths: list[Path]
    color_stream: dict[str, Any]
    depth_stream: dict[str, Any] | None
    color_calibration: dict[str, Any]
    depth_calibration: dict[str, Any] | None
    world_pose: dict[str, Any]
    frame_count: int = 0
    width: int = 0
    height: int = 0


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConversionError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConversionError(f"Invalid JSON in {path}: {exc}") from exc


def discover_project_root(start: Path) -> Path:
    """Accept a project root, dkproject.json, or an ancestor containing one project."""
    start = start.resolve()
    if start.is_file() and start.name.lower() == "dkproject.json":
        return start.parent
    if (start / "dkproject.json").is_file():
        return start
    matches = list(start.rglob("dkproject.json")) if start.is_dir() else []
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise ConversionError(f"Could not find dkproject.json beneath {start}")
    raise ConversionError(
        f"Found {len(matches)} Depthkit projects beneath {start}; pass the intended "
        "project directory explicitly."
    )


def pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64)
    if rotation.shape != (3,) or translation.shape != (3,):
        raise ConversionError("Expected three-element Rodrigues rotation and translation")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
    matrix[:3, 3] = translation
    return matrix


def color_camera_to_world(
    world_pose: dict[str, Any],
    color_extrinsics: dict[str, Any],
    *,
    scatter_basis: str = "scatter",
    color_extrinsics_direction: str = "depth-to-color",
    ignore_color_offset: bool = False,
) -> np.ndarray:
    """Return OpenCV-basis color camera-to-world transform."""
    world_from_depth = pose_matrix(world_pose)
    if scatter_basis == "scatter":
        # Scatter serializes this rig in a left-handed frame. Conjugating by a
        # reflection changes handedness while keeping the rotation proper.
        world_from_depth = SCATTER_HANDEDNESS @ world_from_depth @ SCATTER_HANDEDNESS
    elif scatter_basis == "opengl":
        world_from_depth = world_from_depth @ OPENGL_TO_OPENCV
    elif scatter_basis != "opencv":
        raise ConversionError(f"Unsupported Scatter camera basis: {scatter_basis}")

    if ignore_color_offset:
        return world_from_depth

    _, depth_from_color = color_depth_transforms(
        color_extrinsics, color_extrinsics_direction
    )
    return world_from_depth @ depth_from_color


def color_depth_transforms(
    color_extrinsics: dict[str, Any], direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (color_from_depth, depth_from_color) for either stored convention."""
    stored = pose_matrix(color_extrinsics)
    if direction == "depth-to-color":
        color_from_depth = stored
        depth_from_color = np.linalg.inv(stored)
    elif direction == "color-to-depth":
        depth_from_color = stored
        color_from_depth = np.linalg.inv(stored)
    else:
        raise ConversionError(f"Unsupported color extrinsics direction: {direction}")
    return color_from_depth, depth_from_color


def rotation_matrix_to_colmap_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP's (qw, qx, qy, qz)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    rotation_vector = cv2.Rodrigues(matrix)[0].reshape(3)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    xyz = rotation_vector * (math.sin(angle / 2.0) / angle)
    quat = np.array([math.cos(angle / 2.0), *xyz])
    if quat[0] < 0:
        quat *= -1
    return quat


def intrinsic_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    fx, fy = map(float, intrinsics["focalLength"])
    cx, cy = map(float, intrinsics["principalPoint"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def distortion_vector(intrinsics: dict[str, Any]) -> np.ndarray:
    radial = [float(value) for value in intrinsics.get("distortionRadial", [])]
    tangential = [float(value) for value in intrinsics.get("distortionTangential", [])]
    radial += [0.0] * (6 - len(radial))
    tangential += [0.0] * (2 - len(tangential))
    # OpenCV rational model: k1, k2, p1, p2, k3, k4, k5, k6.
    return np.array(
        [radial[0], radial[1], tangential[0], tangential[1], *radial[2:6]],
        dtype=np.float64,
    )


def stream_of_type(streams: Iterable[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((stream for stream in streams if stream.get("type") == kind), None)


def calibration_for_stream(
    device: dict[str, Any], stream: dict[str, Any], source_type: str
) -> dict[str, Any]:
    source = next(
        (item for item in device.get("sources", []) if item.get("type") == source_type),
        None,
    )
    if source is None:
        raise ConversionError(f"Device {stream.get('deviceConfigurationId')} lacks {source_type}")
    name = stream.get("calibration")
    calibration = source.get("calibrations", {}).get(name)
    if calibration is None:
        raise ConversionError(f"Missing calibration profile {name!r} for {source_type}")
    return calibration


def resolve_asset(project_root: Path, asset_path: str) -> Path:
    return project_root / Path(asset_path.replace("\\", "/"))


def sensor_number_from_asset(asset_path: str) -> int:
    match = SENSOR_NUMBER_RE.search(asset_path)
    if not match:
        raise ConversionError(f"Cannot determine SensorNN index from {asset_path}")
    return int(match.group(1))


def enumerate_depth_files(project_root: Path, asset_path: str) -> list[Path]:
    resolved = resolve_asset(project_root, asset_path)
    if any(token in asset_path for token in DEPTH_PATTERN_TOKENS):
        return sorted(resolved.parent.glob("*.png"))
    if resolved.is_dir():
        return sorted(resolved.glob("*.png"))
    return [resolved] if resolved.is_file() else []


def inspect_video(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ConversionError(f"Cannot open color video (missing, locked, or incomplete): {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ConversionError(f"Color video has invalid metadata: {path}")
    return width, height, frame_count


def gather_cameras(
    project_root: Path, project: dict[str, Any], recording_name: str
) -> tuple[list[CameraInput], list[str]]:
    recording = project.get("recordings", {}).get(recording_name)
    if recording is None:
        names = "\n  ".join(sorted(project.get("recordings", {})))
        raise ConversionError(f"Recording {recording_name!r} is not in dkproject.json. Available:\n  {names}")

    cameras: list[CameraInput] = []
    problems: list[str] = []
    for device_id, streams in recording.get("streams", {}).items():
        color_stream = stream_of_type(streams, "color")
        depth_stream = stream_of_type(streams, "depth")
        if color_stream is None:
            problems.append(f"{device_id}: no color stream in manifest")
            continue
        sensor_number = sensor_number_from_asset(color_stream["assetPath"])
        color_path = resolve_asset(project_root, color_stream["assetPath"])
        depth_paths = (
            enumerate_depth_files(project_root, depth_stream["assetPath"])
            if depth_stream
            else []
        )
        if not color_path.is_file():
            problems.append(f"Sensor{sensor_number:02d} {device_id}: missing color video")
            continue
        try:
            width, height, frame_count = inspect_video(color_path)
        except ConversionError as exc:
            problems.append(str(exc))
            continue

        device = project.get("deviceConfigurations", {}).get(device_id)
        if device is None:
            problems.append(f"{device_id}: missing device calibration")
            continue
        try:
            color_calibration = calibration_for_stream(device, color_stream, "color")
            depth_calibration = (
                calibration_for_stream(device, depth_stream, "depth") if depth_stream else None
            )
            world_pose = device["worldExtrinsics"]["world"]
        except (ConversionError, KeyError) as exc:
            problems.append(f"{device_id}: invalid calibration: {exc}")
            continue
        expected_size = tuple(map(int, color_calibration["intrinsics"]["imageSize"]))
        if (width, height) != expected_size:
            problems.append(
                f"Sensor{sensor_number:02d} {device_id}: video is {width}x{height}, "
                f"calibration is {expected_size[0]}x{expected_size[1]}"
            )
            continue
        cameras.append(
            CameraInput(
                sensor_number=sensor_number,
                camera_index=-1,
                device_id=device_id,
                color_path=color_path,
                depth_paths=depth_paths,
                color_stream=color_stream,
                depth_stream=depth_stream,
                color_calibration=color_calibration,
                depth_calibration=depth_calibration,
                world_pose=world_pose,
                frame_count=frame_count,
                width=width,
                height=height,
            )
        )
    cameras.sort(key=lambda camera: camera.sensor_number)
    for camera_index, camera in enumerate(cameras):
        camera.camera_index = camera_index
    return cameras, problems


def camera_look_at_score(camera_to_world: list[np.ndarray]) -> float:
    """Mean cosine between camera +Z and the rig centroid; near +1 is expected."""
    if len(camera_to_world) < 2:
        return float("nan")
    centers = np.array([matrix[:3, 3] for matrix in camera_to_world])
    centroid = centers.mean(axis=0)
    scores = []
    for matrix, center in zip(camera_to_world, centers):
        direction = centroid - center
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            scores.append(float(np.dot(matrix[:3, 2], direction / norm)))
    return float(np.mean(scores)) if scores else float("nan")


def prepare_camera_models(
    cameras: list[CameraInput], args: argparse.Namespace
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    new_intrinsics: list[np.ndarray] = []
    distortions: list[np.ndarray] = []
    color_to_world: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for camera in cameras:
        intrinsics = camera.color_calibration["intrinsics"]
        old_k = intrinsic_matrix(intrinsics)
        distortion = distortion_vector(intrinsics)
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            old_k, distortion, (camera.width, camera.height), args.alpha, (camera.width, camera.height)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            old_k,
            distortion,
            None,
            new_k,
            (camera.width, camera.height),
            cv2.CV_32FC1,
        )
        c2w = color_camera_to_world(
            camera.world_pose,
            camera.color_calibration["extrinsics"],
            scatter_basis=args.scatter_basis,
            color_extrinsics_direction=args.color_extrinsics_direction,
            ignore_color_offset=args.ignore_color_offset,
        )
        new_intrinsics.append(new_k)
        distortions.append(distortion)
        color_to_world.append(c2w)
        maps.append((map_x, map_y))
    return new_intrinsics, distortions, color_to_world, maps


def write_colmap_cameras(path: Path, cameras: list[CameraInput], matrices: list[np.ndarray]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera, matrix in zip(cameras, matrices):
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        lines.append(
            f"{camera.camera_index + 1} PINHOLE {camera.width} {camera.height} "
            f"{fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_colmap_images(path: Path, cameras: list[CameraInput], c2w: list[np.ndarray]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "# POINTS2D[] is intentionally empty for fixed-rig 4C4D input",
    ]
    for camera, world_from_color in zip(cameras, c2w):
        color_from_world = np.linalg.inv(world_from_color)
        quaternion = rotation_matrix_to_colmap_quaternion(color_from_world[:3, :3])
        translation = color_from_world[:3, 3]
        values = [*quaternion, *translation]
        lines.append(
            f"{camera.camera_index + 1} "
            + " ".join(f"{value:.12g}" for value in values)
            + f" {camera.camera_index + 1} cam{camera.camera_index:02d}_0000.png"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_frames(
    cameras: list[CameraInput],
    maps: list[tuple[np.ndarray, np.ndarray]],
    image_dir: Path,
    count: int,
    start_frame: int,
) -> None:
    for camera, (map_x, map_y) in zip(cameras, maps):
        capture = cv2.VideoCapture(str(camera.color_path))
        if not capture.isOpened():
            raise ConversionError(f"Could not reopen {camera.color_path}")
        try:
            if start_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_index in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ConversionError(
                        f"Decode stopped at frame {frame_index} for {camera.color_path}"
                    )
                undistorted = cv2.remap(
                    frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                output = image_dir / f"cam{camera.camera_index:02d}_{frame_index:04d}.png"
                if not cv2.imwrite(str(output), undistorted):
                    raise ConversionError(f"Failed to write {output}")
        finally:
            capture.release()


def validate_output_path(output: Path, project_root: Path) -> None:
    """Reject broad overwrite targets that could contain source or workspace data."""
    output = output.resolve()
    project_root = project_root.resolve()
    cwd = Path.cwd().resolve()
    if output.parent == output:
        raise ConversionError(f"Refusing to use a filesystem root as output: {output}")
    if output in (project_root, cwd):
        raise ConversionError(f"Refusing to overwrite project/workspace directory: {output}")
    if project_root.is_relative_to(output):
        raise ConversionError(f"Output cannot be an ancestor of the source project: {output}")


def unproject_depth_points(
    camera: CameraInput,
    color_k: np.ndarray,
    world_from_color: np.ndarray,
    undistorted_color: np.ndarray,
    *,
    depth_index: int,
    stride: int,
    max_depth: float,
    color_extrinsics_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not camera.depth_paths or camera.depth_calibration is None:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    if depth_index >= len(camera.depth_paths):
        raise ConversionError(
            f"Depth stream for Sensor{camera.sensor_number:02d} has no frame {depth_index}"
        )
    depth_path = camera.depth_paths[depth_index]
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ConversionError(f"Could not read 16-bit depth image {depth_path}")
    depth_intrinsics = camera.depth_calibration["intrinsics"]
    expected = tuple(map(int, depth_intrinsics["imageSize"]))
    if (depth.shape[1], depth.shape[0]) != expected:
        raise ConversionError(
            f"Depth image {depth_path} is {depth.shape[1]}x{depth.shape[0]}, "
            f"expected {expected[0]}x{expected[1]}"
        )
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride].astype(np.float64) * 0.001  # Depthkit PNG depth is mm.
    valid = (z > 0.0) & (z <= max_depth)
    pixels = np.stack([xs[valid], ys[valid]], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    normalized = cv2.undistortPoints(
        pixels, intrinsic_matrix(depth_intrinsics), distortion_vector(depth_intrinsics)
    ).reshape(-1, 2)
    z_valid = z[valid]
    depth_points = np.column_stack(
        [normalized[:, 0] * z_valid, normalized[:, 1] * z_valid, z_valid]
    )

    color_from_depth, _ = color_depth_transforms(
        camera.color_calibration["extrinsics"], color_extrinsics_direction
    )
    homogeneous_depth = np.column_stack([depth_points, np.ones(len(depth_points))])
    color_points = (color_from_depth @ homogeneous_depth.T).T[:, :3]
    in_front = color_points[:, 2] > 1e-6
    color_points = color_points[in_front]
    depth_points = depth_points[in_front]
    uvw = (color_k @ color_points.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    inside = (
        (u >= 0)
        & (u < undistorted_color.shape[1])
        & (v >= 0)
        & (v < undistorted_color.shape[0])
    )
    colors_bgr = undistorted_color[v[inside], u[inside]]
    depth_points = depth_points[inside]

    # Reconstruct depth-camera c2w from color c2w and the factory depth->color transform.
    world_from_depth = world_from_color @ color_from_depth
    world_points = (
        world_from_depth @ np.column_stack([depth_points, np.ones(len(depth_points))]).T
    ).T[:, :3]
    return world_points, colors_bgr[:, ::-1]


def write_initial_points(
    path: Path,
    cameras: list[CameraInput],
    color_matrices: list[np.ndarray],
    color_to_world: list[np.ndarray],
    image_dir: Path,
    args: argparse.Namespace,
) -> int:
    point_id = 1
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list generated from frame-zero Depthkit depth maps\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for camera, matrix, c2w in zip(cameras, color_matrices, color_to_world):
            color = cv2.imread(str(image_dir / f"cam{camera.camera_index:02d}_0000.png"))
            if color is None:
                raise ConversionError(f"Missing frame-zero color image for camera {camera.camera_index}")
            points, colors = unproject_depth_points(
                camera,
                matrix,
                c2w,
                color,
                depth_index=args.start_frame,
                stride=args.depth_stride,
                max_depth=args.max_depth,
                color_extrinsics_direction=args.color_extrinsics_direction,
            )
            for point, rgb in zip(points, colors):
                handle.write(
                    f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                    f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n"
                )
                point_id += 1
    return point_id - 1


def print_validation(
    recording_name: str,
    cameras: list[CameraInput],
    problems: list[str],
    look_score: float,
) -> None:
    print(f"Recording: {recording_name}")
    print(f"Usable calibrated color cameras: {len(cameras)}")
    print(f"Rig look-at score: {look_score:.4f} (expected close to +1.0)")
    print("\nCamera inventory:")
    for camera in cameras:
        depth = len(camera.depth_paths)
        delta = depth - camera.frame_count if depth else None
        depth_text = "missing" if not depth else f"{depth} frames (delta {delta:+d})"
        print(
            f"  cam{camera.camera_index:02d} <- Sensor{camera.sensor_number:02d} "
            f"{camera.device_id}: color={camera.frame_count} frames, depth={depth_text}"
        )
    if problems:
        print("\nMissing/incomplete sensors:")
        for problem in problems:
            print(f"  - {problem}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Depthkit project root, ancestor, or dkproject.json")
    parser.add_argument("recording", help="Recording key/take directory name from dkproject.json")
    parser.add_argument("output", type=Path, nargs="?", help="4C4D scene output directory")
    parser.add_argument("--validate-only", action="store_true", help="Inspect inputs without writing output")
    parser.add_argument("--allow-incomplete", action="store_true", help="Convert available cameras only")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--max-frames", type=int, help="Limit converted timeline frames")
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Source RGB/depth frame that becomes output frame 0000",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV undistort alpha, 0=crop, 1=retain FOV")
    point_group = parser.add_mutually_exclusive_group()
    point_group.add_argument(
        "--depth-points",
        action="store_true",
        help="Opt in to initializing points3D.txt from depth PNGs (RGB-only is the default)",
    )
    point_group.add_argument(
        "--no-points",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--depth-stride", type=int, default=4, help="Depth sampling stride for initialization")
    parser.add_argument("--max-depth", type=float, default=6.0, help="Maximum initialization depth in meters")
    parser.add_argument(
        "--scatter-basis", choices=("scatter", "opengl", "opencv"), default="scatter",
        help="Basis used by worldExtrinsics (default: Scatter left-handed serialization)",
    )
    parser.add_argument(
        "--color-extrinsics-direction",
        choices=("depth-to-color", "color-to-depth"),
        default="depth-to-color",
    )
    parser.add_argument("--ignore-color-offset", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 0.0 <= args.alpha <= 1.0:
            raise ConversionError("--alpha must be between 0 and 1")
        if args.depth_stride <= 0:
            raise ConversionError("--depth-stride must be positive")
        if args.start_frame < 0:
            raise ConversionError("--start-frame cannot be negative")
        if args.max_depth <= 0:
            raise ConversionError("--max-depth must be positive")
        project_root = discover_project_root(args.project)
        project = load_json(project_root / "dkproject.json")
        cameras, problems = gather_cameras(project_root, project, args.recording)
        if not cameras:
            raise ConversionError("No complete calibrated color cameras are currently readable")
        color_k, _, color_to_world, maps = prepare_camera_models(cameras, args)
        look_score = camera_look_at_score(color_to_world)
        print_validation(args.recording, cameras, problems, look_score)

        if look_score < 0.7:
            raise ConversionError(
                "Converted optical axes do not converge on the rig center. Check --scatter-basis "
                "and pose assumptions before emitting COLMAP files."
            )
        expected_cameras = len(project["recordings"][args.recording].get("streams", {}))
        if args.validate_only:
            return 0 if not problems and len(cameras) == expected_cameras else 2
        if not args.allow_incomplete and len(cameras) != expected_cameras:
            raise ConversionError(
                f"Only {len(cameras)}/{expected_cameras} cameras are ready. Rerun after copying, "
                "or use --allow-incomplete intentionally."
            )
        if len(cameras) < 4 and not args.validate_only:
            raise ConversionError("4C4D conversion requires at least four usable color cameras")
        if args.output is None:
            raise ConversionError("OUTPUT is required unless --validate-only is used")

        output = args.output.resolve()
        validate_output_path(output, project_root)
        if output.exists():
            if not args.overwrite:
                raise ConversionError(f"Output already exists: {output}; use --overwrite")
            shutil.rmtree(output)
        image_dir = output / "images"
        sparse_dir = output / "sparse" / "0"
        image_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)

        frame_count = min(camera.frame_count - args.start_frame for camera in cameras)
        if frame_count <= 0:
            raise ConversionError(
                f"--start-frame {args.start_frame} is beyond the common RGB timeline"
            )
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ConversionError("--max-frames must be positive")
            frame_count = min(frame_count, args.max_frames)
        if frame_count > 10000:
            raise ConversionError("4C4D's four-digit timestamp parser supports at most 10,000 frames")

        write_colmap_cameras(sparse_dir / "cameras.txt", cameras, color_k)
        write_colmap_images(sparse_dir / "images.txt", cameras, color_to_world)
        print(f"\nExtracting and undistorting {frame_count} frames per camera...")
        extract_frames(cameras, maps, image_dir, frame_count, args.start_frame)
        if args.depth_points:
            point_count = write_initial_points(
                sparse_dir / "points3D.txt", cameras, color_k, color_to_world, image_dir, args
            )
            point_source = "depth"
        else:
            (sparse_dir / "points3D.txt").write_text(
                "# Empty RGB-only initialization; generate points with COLMAP/MASt3R/MAtCha\n",
                encoding="utf-8",
            )
            point_count = 0
            point_source = "none"

        manifest = {
            "sourceProject": str(project_root),
            "recording": args.recording,
            "frameCount": frame_count,
            "sourceStartFrame": args.start_frame,
            "cameraCount": len(cameras),
            "initialPointCount": point_count,
            "initialPointSource": point_source,
            "fps": 30,
            "poseAssumptions": {
                "worldExtrinsicsDirection": "camera-to-world",
                "scatterCameraBasis": args.scatter_basis,
                "colorExtrinsicsDirection": args.color_extrinsics_direction,
                "ignoreColorOffset": args.ignore_color_offset,
                "colmapOutput": "OpenCV world-to-camera",
            },
            "rigLookAtScore": look_score,
            "undistortAlpha": args.alpha,
            "cameras": [
                {
                    "cameraIndex": camera.camera_index,
                    "sensorNumber": camera.sensor_number,
                    "deviceId": camera.device_id,
                    "sourceColor": str(camera.color_path),
                    "sourceDepthFrames": len(camera.depth_paths),
                }
                for camera in cameras
            ],
            "warnings": problems,
        }
        (output / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote 4C4D dataset to {output}")
        print(f"Initial points: {point_count}")
        return 0
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
