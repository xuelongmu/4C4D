"""Interactive browser viewer for trained 4C4D checkpoints.

The browser UI is provided by Viser, while all pixels are produced by the
repository's native 4D CUDA rasterizer.  This keeps the temporal Gaussian
conditioning and temporal spherical harmonics identical to render.py.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import math
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
MIN_FOCAL_LENGTH_MM = 1.0
MAX_FOCAL_LENGTH_MM = 500.0


def quaternion_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion in WXYZ order to a 3x3 rotation matrix."""
    q = np.asarray(wxyz, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in WXYZ order."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def opengl_c2w_to_opencv_c2w(wxyz: np.ndarray, position: np.ndarray) -> np.ndarray:
    c2w_gl = np.eye(4, dtype=np.float64)
    c2w_gl[:3, :3] = quaternion_wxyz_to_matrix(wxyz)
    c2w_gl[:3, 3] = np.asarray(position, dtype=np.float64)
    return c2w_gl @ OPENGL_TO_OPENCV


def opencv_c2w_to_viser(c2w_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w_gl = np.asarray(c2w_cv, dtype=np.float64) @ OPENGL_TO_OPENCV
    return matrix_to_quaternion_wxyz(c2w_gl[:3, :3]), c2w_gl[:3, 3].copy()


@dataclass(frozen=True)
class CameraSnapshot:
    wxyz: np.ndarray
    position: np.ndarray
    fov_y: float
    aspect: float
    canvas_width: int
    canvas_height: int
    frame: float
    render_width: int
    shot_aspect: float | None = None
    viewport_aspect: float | None = None
    preview_framing: bool = False


@dataclass
class ShotKeyframe:
    """A cinematic camera and lens pose captured on the shot timeline."""

    shot_frame: int
    wxyz: np.ndarray
    position: np.ndarray
    fov_y: float


def quaternion_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path spherical interpolation for WXYZ quaternions."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        result = qa + t * (qb - qa)
        return result / np.linalg.norm(result)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - t) * theta) / sin_theta) * qa + (math.sin(t * theta) / sin_theta) * qb


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Uniform Catmull-Rom interpolation used by the cinematic camera path."""
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def interpolate_keyframes(keyframes: list[ShotKeyframe], shot_frame: float, smooth: bool) -> ShotKeyframe:
    """Evaluate a camera shot, clamping before its first and after its last key."""
    if not keyframes:
        raise ValueError("A shot needs at least one keyframe")
    ordered = sorted(keyframes, key=lambda key: key.shot_frame)
    if shot_frame <= ordered[0].shot_frame:
        return ordered[0]
    if shot_frame >= ordered[-1].shot_frame:
        return ordered[-1]
    left, right = ordered[0], ordered[-1]
    segment_index = 0
    for index in range(len(ordered) - 1):
        if ordered[index].shot_frame <= shot_frame <= ordered[index + 1].shot_frame:
            left, right = ordered[index], ordered[index + 1]
            segment_index = index
            break
    alpha = (shot_frame - left.shot_frame) / max(right.shot_frame - left.shot_frame, 1)
    position = (1.0 - alpha) * left.position + alpha * right.position
    if smooth:
        before = ordered[max(0, segment_index - 1)]
        after = ordered[min(len(ordered) - 1, segment_index + 2)]
        position = catmull_rom(before.position, left.position, right.position, after.position, alpha)
        alpha = smoothstep(alpha)
    return ShotKeyframe(
        shot_frame=int(round(shot_frame)),
        wxyz=quaternion_slerp(left.wxyz, right.wxyz, alpha),
        position=position,
        fov_y=(1.0 - alpha) * left.fov_y + alpha * right.fov_y,
    )


def scene_frame_for_shot(
    shot_frame: float,
    shot_fps: float,
    num_scene_frames: int,
    time_duration: tuple[float, float],
) -> float:
    """Evaluate dynamic-scene time from the sequencer's elapsed time."""
    duration_seconds = max(float(time_duration[1] - time_duration[0]), 1e-6)
    native_scene_fps = float(num_scene_frames) / duration_seconds
    scene_frame = float(shot_frame) / max(float(shot_fps), 1.0) * native_scene_fps
    return float(np.clip(scene_frame, 0.0, max(num_scene_frames - 1, 0)))


def focal_length_to_fov_y(focal_length_mm: float, sensor_height_mm: float) -> float:
    return 2.0 * math.atan(max(sensor_height_mm, 1e-6) / (2.0 * max(focal_length_mm, 1e-6)))


def fov_y_to_focal_length(fov_y: float, sensor_height_mm: float) -> float:
    return max(sensor_height_mm, 1e-6) / (2.0 * math.tan(max(fov_y, 1e-6) * 0.5))


def shot_gate_dimensions(sensor_width_mm: float, sensor_height_mm: float, shot_aspect: float) -> tuple[float, float]:
    """Return the centered crop of a physical filmback used by the shot gate."""
    sensor_width_mm = max(float(sensor_width_mm), 1e-6)
    sensor_height_mm = max(float(sensor_height_mm), 1e-6)
    shot_aspect = max(float(shot_aspect), 1e-6)
    if shot_aspect >= sensor_width_mm / sensor_height_mm:
        return sensor_width_mm, sensor_width_mm / shot_aspect
    return sensor_height_mm * shot_aspect, sensor_height_mm


def focal_length_to_shot_fov_y(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
) -> float:
    return focal_length_to_fov_y(
        focal_length_mm,
        shot_gate_dimensions(sensor_width_mm, sensor_height_mm, shot_aspect)[1],
    )


def shot_fov_y_to_focal_length(
    fov_y: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
) -> float:
    return fov_y_to_focal_length(
        fov_y,
        shot_gate_dimensions(sensor_width_mm, sensor_height_mm, shot_aspect)[1],
    )


def preview_fov_y_for_gate(shot_fov_y: float, viewport_aspect: float, shot_aspect: float) -> float:
    """Expand vertical FOV when a wide shot gate is letterboxed in a narrow viewport."""
    viewport_aspect = max(float(viewport_aspect), 1e-6)
    shot_aspect = max(float(shot_aspect), 1e-6)
    if viewport_aspect >= shot_aspect:
        return float(shot_fov_y)
    return 2.0 * math.atan(math.tan(float(shot_fov_y) * 0.5) * shot_aspect / viewport_aspect)


def export_keyframes(
    keyframes: list[ShotKeyframe],
    duration_frames: int,
    smooth: bool,
    bake_every_frame: bool = False,
) -> list[ShotKeyframe]:
    """Bake smooth motion per frame and ensure every sidecar reaches the shot tail."""
    duration_frames = max(int(duration_frames), 1)
    if smooth or bake_every_frame:
        baked: list[ShotKeyframe] = []
        for frame in range(duration_frames):
            key = interpolate_keyframes(keyframes, frame, smooth=smooth)
            baked.append(ShotKeyframe(frame, key.wxyz.copy(), key.position.copy(), key.fov_y))
        return baked
    ordered = [
        ShotKeyframe(key.shot_frame, key.wxyz.copy(), key.position.copy(), key.fov_y)
        for key in sorted(keyframes, key=lambda item: item.shot_frame)
        if key.shot_frame < duration_frames
    ]
    if not ordered or ordered[0].shot_frame > 0:
        head = interpolate_keyframes(keyframes, 0, smooth=False)
        ordered.insert(0, ShotKeyframe(0, head.wxyz.copy(), head.position.copy(), head.fov_y))
    tail_frame = duration_frames - 1
    if ordered[-1].shot_frame < tail_frame:
        tail = interpolate_keyframes(keyframes, tail_frame, smooth=False)
        ordered.append(ShotKeyframe(tail_frame, tail.wxyz.copy(), tail.position.copy(), tail.fov_y))
    return ordered


def focal_length_to_full_frame_equivalent(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
) -> float:
    sensor_gate_width, _ = shot_gate_dimensions(
        sensor_width_mm, sensor_height_mm, shot_aspect
    )
    full_frame_gate_width, _ = shot_gate_dimensions(36.0, 24.0, shot_aspect)
    return float(focal_length_mm) * full_frame_gate_width / sensor_gate_width


def full_frame_equivalent_to_focal_length(
    equivalent_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
) -> float:
    sensor_gate_width, _ = shot_gate_dimensions(
        sensor_width_mm, sensor_height_mm, shot_aspect
    )
    full_frame_gate_width, _ = shot_gate_dimensions(36.0, 24.0, shot_aspect)
    return float(equivalent_mm) * sensor_gate_width / full_frame_gate_width


def rotation_matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ Euler angles in degrees for precise camera editing."""
    m = np.asarray(matrix, dtype=np.float64)
    sy = math.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    if sy > 1e-7:
        x = math.atan2(m[2, 1], m[2, 2])
        y = math.atan2(-m[2, 0], sy)
        z = math.atan2(m[1, 0], m[0, 0])
    else:
        x = math.atan2(-m[1, 2], m[1, 1])
        y = math.atan2(-m[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def euler_xyz_degrees_to_quaternion(euler_degrees: np.ndarray) -> np.ndarray:
    x, y, z = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    return matrix_to_quaternion_wxyz(rotation)


def shot_to_json_bytes(
    name: str,
    fps: int,
    keyframes: list[ShotKeyframe],
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
    duration_frames: int | None = None,
) -> bytes:
    payload = {
        "schema": "4c4d.camera-shot/1.0",
        "name": name,
        "fps": fps,
        "coordinate_system": "right-handed, camera looks down -Z, WXYZ quaternions",
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "shot_aspect_ratio": shot_aspect,
        "duration_frames": duration_frames or max(key.shot_frame for key in keyframes) + 1,
        "keyframes": [
            {
                "shot_frame": key.shot_frame,
                "position": key.position.tolist(),
                "rotation_wxyz": key.wxyz.tolist(),
                "vertical_fov_degrees": math.degrees(key.fov_y),
                "focal_length_mm": shot_fov_y_to_focal_length(
                    key.fov_y, sensor_width_mm, sensor_height_mm, shot_aspect
                ),
                "horizontal_fov_degrees": math.degrees(
                    2.0 * math.atan(math.tan(key.fov_y * 0.5) * shot_aspect)
                ),
            }
            for key in sorted(keyframes, key=lambda item: item.shot_frame)
        ],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def shot_to_gltf_bytes(
    name: str,
    fps: int,
    keyframes: list[ShotKeyframe],
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
    duration_frames: int | None = None,
) -> bytes:
    """Create an embedded glTF 2.0 camera animation without optional dependencies."""
    ordered = sorted(keyframes, key=lambda key: key.shot_frame)
    times = np.asarray([key.shot_frame / fps for key in ordered], dtype="<f4")
    translations = np.asarray([key.position for key in ordered], dtype="<f4")
    rotations = np.asarray([[key.wxyz[1], key.wxyz[2], key.wxyz[3], key.wxyz[0]] for key in ordered], dtype="<f4")
    vertical_fovs = np.asarray([key.fov_y for key in ordered], dtype="<f4")
    chunks: list[bytes] = []
    views: list[dict[str, int]] = []
    offset = 0
    for array in (times, translations, rotations, vertical_fovs):
        raw = array.tobytes()
        padding = (-len(raw)) % 4
        chunks.append(raw + b"\0" * padding)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        offset += len(raw) + padding
    binary = b"".join(chunks)
    first = ordered[0]
    gltf = {
        "asset": {"version": "2.0", "generator": "4C4D Cinematic Viewer"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "4C4D_Z_up_to_glTF_Y_up", "rotation": [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], "children": [1]},
            {"name": name, "camera": 0},
        ],
        "cameras": [{"name": name, "type": "perspective", "perspective": {"yfov": first.fov_y, "aspectRatio": shot_aspect, "znear": 0.01, "zfar": 100.0}}],
        "buffers": [{"byteLength": len(binary), "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode("ascii")}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(ordered), "type": "SCALAR", "min": [float(times.min())], "max": [float(times.max())]},
            {"bufferView": 1, "componentType": 5126, "count": len(ordered), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(ordered), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": len(ordered), "type": "SCALAR"},
        ],
        "extensionsUsed": ["KHR_animation_pointer"],
        "animations": [{"name": name, "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}, {"input": 0, "output": 2, "interpolation": "LINEAR"}, {"input": 0, "output": 3, "interpolation": "LINEAR"}], "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}, {"sampler": 1, "target": {"node": 1, "path": "rotation"}}, {"sampler": 2, "target": {"extensions": {"KHR_animation_pointer": {"pointer": "/cameras/0/perspective/yfov"}}}}]}],
        "extras": {
            "fps": fps,
            "shot_aspect_ratio": shot_aspect,
            "sensor_width_mm": sensor_width_mm,
            "sensor_height_mm": sensor_height_mm,
            "duration_frames": duration_frames or ordered[-1].shot_frame + 1,
            "focal_length_mm": [
                shot_fov_y_to_focal_length(key.fov_y, sensor_width_mm, sensor_height_mm, shot_aspect)
                for key in ordered
            ],
        },
    }
    return json.dumps(gltf, indent=2).encode("utf-8")


def shot_to_usda_bytes(
    name: str,
    fps: int,
    keyframes: list[ShotKeyframe],
    sensor_width_mm: float,
    sensor_height_mm: float,
    shot_aspect: float,
    duration_frames: int | None = None,
) -> bytes:
    ordered = sorted(keyframes, key=lambda key: key.shot_frame)
    gate_width_mm, gate_height_mm = shot_gate_dimensions(
        sensor_width_mm, sensor_height_mm, shot_aspect
    )
    translation_samples: list[str] = []
    orientation_samples: list[str] = []
    focal_samples: list[str] = []
    for key in ordered:
        position = ", ".join(f"{value:.9g}" for value in key.position)
        quaternion = np.asarray(key.wxyz, dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        imaginary = ", ".join(f"{value:.9g}" for value in quaternion[1:])
        translation_samples.append(f"            {key.shot_frame}: ({position})")
        orientation_samples.append(
            f"            {key.shot_frame}: ({quaternion[0]:.9g}, ({imaginary}))"
        )
        focal_samples.append(
            f"            {key.shot_frame}: {shot_fov_y_to_focal_length(key.fov_y, sensor_width_mm, sensor_height_mm, shot_aspect):.9g}"
        )
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name) or "CameraShot"
    if not re.match(r"[A-Za-z_]", safe_name[0]):
        safe_name = "_" + safe_name
    translation_sample_text = ",\n".join(translation_samples)
    orientation_sample_text = ",\n".join(orientation_samples)
    focal_sample_text = ",\n".join(focal_samples)
    text = f'''#usda 1.0
(
    defaultPrim = "{safe_name}"
    startTimeCode = {ordered[0].shot_frame}
    endTimeCode = {(duration_frames or ordered[-1].shot_frame + 1) - 1}
    timeCodesPerSecond = {fps}
    upAxis = "Z"
)

def Camera "{safe_name}"
{{
    double3 xformOp:translate.timeSamples = {{
{translation_sample_text}
    }}
    quatd xformOp:orient.timeSamples = {{
{orientation_sample_text}
    }}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    float focalLength.timeSamples = {{
{focal_sample_text}
    }}
    float verticalAperture = {gate_height_mm}
    float horizontalAperture = {gate_width_mm}
    custom double physicalSensorWidthMm = {sensor_width_mm}
    custom double physicalSensorHeightMm = {sensor_height_mm}
    custom double shotAspectRatio = {shot_aspect}
}}
'''
    return text.encode("utf-8")


def compose_framed_preview(
    image: np.ndarray,
    viewport_aspect: float,
    shot_aspect: float,
    matte_opacity: float,
    rule_of_thirds: bool,
    action_safe: bool,
) -> np.ndarray:
    """Fit an exact shot gate into the viewport and darken visible overscan."""
    source_height, source_width = image.shape[:2]
    viewport_aspect = max(float(viewport_aspect), 1e-6)
    shot_aspect = max(float(shot_aspect), 1e-6)

    # Render the full viewport as overscan, then place the exact shot gate
    # inside it. This preserves visible context beneath the matte for both
    # pillarboxed and letterboxed framing.
    framed = image.copy()
    output_height, output_width = framed.shape[:2]
    opacity = float(np.clip(matte_opacity, 0.0, 1.0))
    if viewport_aspect + 1e-6 < shot_aspect:
        frame_height = min(output_height, int(round(output_width / shot_aspect)))
        y0 = (output_height - frame_height) // 2
        frame_rect = (0, y0, output_width, y0 + frame_height)
        if y0 > 0 and opacity > 0.0:
            framed[:y0] = np.rint(framed[:y0].astype(np.float32) * (1.0 - opacity)).astype(np.uint8)
            framed[y0 + frame_height :] = np.rint(
                framed[y0 + frame_height :].astype(np.float32) * (1.0 - opacity)
            ).astype(np.uint8)
    else:
        frame_width = min(output_width, int(round(output_height * shot_aspect)))
        x0 = (output_width - frame_width) // 2
        frame_rect = (x0, 0, x0 + frame_width, output_height)
        if x0 > 0 and opacity > 0.0:
            framed[:, :x0] = np.rint(framed[:, :x0].astype(np.float32) * (1.0 - opacity)).astype(np.uint8)
            framed[:, x0 + frame_width :] = np.rint(
                framed[:, x0 + frame_width :].astype(np.float32) * (1.0 - opacity)
            ).astype(np.uint8)

    x0, y0, x1, y1 = frame_rect
    color = np.array([235, 235, 235], dtype=np.uint8)
    line = max(1, int(round(min(framed.shape[:2]) / 420.0)))

    def draw_rectangle(left: int, top: int, right: int, bottom: int) -> None:
        left, right = max(0, left), min(framed.shape[1], right)
        top, bottom = max(0, top), min(framed.shape[0], bottom)
        framed[top : min(bottom, top + line), left:right] = color
        framed[max(top, bottom - line) : bottom, left:right] = color
        framed[top:bottom, left : min(right, left + line)] = color
        framed[top:bottom, max(left, right - line) : right] = color

    if rule_of_thirds:
        for x in (x0 + (x1 - x0) // 3, x0 + 2 * (x1 - x0) // 3):
            framed[y0:y1, max(x0, x - line // 2) : min(x1, x + line)] = color
        for y in (y0 + (y1 - y0) // 3, y0 + 2 * (y1 - y0) // 3):
            framed[max(y0, y - line // 2) : min(y1, y + line), x0:x1] = color
    if action_safe:
        safe_x = int((x1 - x0) * 0.05)
        safe_y = int((y1 - y0) * 0.05)
        draw_rectangle(x0 + safe_x, y0 + safe_y, x1 - safe_x, y1 - safe_y)
    return framed


@dataclass
class ReferenceCamera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    c2w_cv: np.ndarray

    @property
    def fov_y(self) -> float:
        return 2.0 * math.atan2(self.height * 0.5, self.fy)

    def viser_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return opencv_c2w_to_viser(self.c2w_cv)


class RenderCamera:
    """Minimal camera interface consumed by gaussian_renderer.render()."""

    def __init__(self, snapshot: CameraSnapshot, timestamp: float, device: str = "cuda") -> None:
        import torch
        from utils.graphics_utils import getProjectionMatrix

        width = max(64, int(snapshot.render_width))
        aspect = max(float(snapshot.aspect), 1e-6)
        height = max(64, int(round(width / aspect)))
        width -= width % 2
        height -= height % 2

        self.image_width = width
        self.image_height = height
        self.FoVy = float(snapshot.fov_y)
        self.FoVx = 2.0 * math.atan(math.tan(self.FoVy * 0.5) * aspect)
        self.znear = 0.01
        self.zfar = 100.0
        self.timestamp = float(timestamp)

        c2w_cv = opengl_c2w_to_opencv_c2w(snapshot.wxyz, snapshot.position)
        w2c = np.linalg.inv(c2w_cv)
        self.world_view_transform = torch.as_tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(device)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_reference_cameras(path: Path, training_views: list[int]) -> dict[str, ReferenceCamera]:
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    wanted = {f"cam{view:02d}" for view in training_views}
    references: dict[str, ReferenceCamera] = {}
    for entry in entries:
        image_name = str(entry["img_name"])
        view_name, _, frame_text = image_name.rpartition("_")
        if view_name not in wanted or frame_text != "0000" or view_name in references:
            continue
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.asarray(entry["rotation"], dtype=np.float64)
        c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
        references[view_name] = ReferenceCamera(
            name=view_name,
            width=int(entry["width"]),
            height=int(entry["height"]),
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            c2w_cv=c2w,
        )
    if not references:
        raise RuntimeError(f"No requested training cameras found in {path}")
    return references


def load_model(config_path: Path, checkpoint_path: Path) -> tuple[Any, Any, int, tuple[float, float]]:
    """Load inference tensors only; optimizer state remains on CPU and is discarded."""
    import torch
    from omegaconf import OmegaConf
    from scene.gaussian_model import GaussianModel

    cfg = OmegaConf.load(config_path)
    time_duration = (float(cfg.time_duration[0]), float(cfg.time_duration[1]))
    sh_degree_t = 2 if bool(cfg.PipelineParams.eval_shfs_4d) else 0
    model = GaussianModel(
        int(cfg.ModelParams.sh_degree),
        gaussian_dim=int(cfg.gaussian_dim),
        time_duration=list(time_duration),
        rot_4d=bool(cfg.rot_4d),
        force_sh_3d=bool(cfg.force_sh_3d),
        sh_degree_t=sh_degree_t,
    )

    print(f"Loading checkpoint on CPU: {checkpoint_path}", flush=True)
    model_args, iteration = torch.load(checkpoint_path, map_location="cpu")
    if len(model_args) != 21:
        raise RuntimeError(f"Expected a 21-field 4D checkpoint capture, found {len(model_args)} fields")

    tensor_fields = {
        "_xyz": 1,
        "_features_dc": 2,
        "_features_rest": 3,
        "_scaling": 4,
        "_rotation": 5,
        "_opacity": 6,
        "_t": 14,
        "_scaling_t": 15,
        "_rotation_r": 16,
        "env_map": 18,
    }
    model.active_sh_degree = int(model_args[0])
    model.spatial_lr_scale = float(model_args[13])
    model.rot_4d = bool(model_args[17])
    model.active_sh_degree_t = int(model_args[19])
    for field, index in tensor_fields.items():
        value = model_args[index]
        if isinstance(value, torch.Tensor):
            setattr(model, field, value.detach().to("cuda", non_blocking=False).contiguous())

    del model_args
    gc.collect()
    torch.cuda.empty_cache()

    pipe = SimpleNamespace(
        convert_SHs_python=bool(cfg.PipelineParams.convert_SHs_python),
        compute_cov3D_python=bool(cfg.PipelineParams.compute_cov3D_python),
        debug=False,
        env_map_res=int(cfg.PipelineParams.env_map_res),
    )
    print(
        f"Loaded iteration {iteration:,}: {model.get_xyz.shape[0]:,} Gaussians, "
        f"spatial SH {model.active_sh_degree}, temporal SH {model.active_sh_degree_t}",
        flush=True,
    )
    return model, pipe, int(iteration), time_duration


def frame_to_timestamp(frame: int, num_frames: int, time_duration: tuple[float, float]) -> float:
    start, end = time_duration
    return start + (end - start) * float(frame) / float(num_frames)


def render_image(model: Any, pipe: Any, snapshot: CameraSnapshot, num_frames: int, time_duration: tuple[float, float], white_background: bool) -> tuple[np.ndarray, float]:
    import torch
    from gaussian_renderer import render

    timestamp = frame_to_timestamp(snapshot.frame, num_frames, time_duration)
    camera = RenderCamera(snapshot, timestamp)
    background = torch.ones(3, dtype=torch.float32, device="cuda") if white_background else torch.zeros(3, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = render(camera, model, pipe, background)
        image = result["render"].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    torch.cuda.synchronize()
    return image, time.perf_counter() - started


def render_shot_mp4(
    model: Any,
    pipe: Any,
    keyframes: list[ShotKeyframe],
    duration_frames: int,
    fps: int,
    width: int,
    aspect: float,
    interpolation: str,
    num_scene_frames: int,
    time_duration: tuple[float, float],
    white_background: bool,
    crf: int,
    output_path: Path,
    progress_callback: Any,
    cancelled: threading.Event,
) -> None:
    """Render an interpolated camera move and stream RGB frames into ffmpeg."""
    width -= width % 2
    height = max(64, int(round(width / max(aspect, 1e-6))))
    height -= height % 2
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for shot_frame in range(duration_frames):
            if cancelled.is_set():
                raise InterruptedError("Render cancelled")
            key = interpolate_keyframes(keyframes, shot_frame, smooth=interpolation == "Smooth ease")
            snapshot = CameraSnapshot(
                wxyz=key.wxyz,
                position=key.position,
                fov_y=key.fov_y,
                aspect=aspect,
                canvas_width=width,
                canvas_height=height,
                frame=scene_frame_for_shot(shot_frame, fps, num_scene_frames, time_duration),
                render_width=width,
            )
            image, _ = render_image(model, pipe, snapshot, num_scene_frames, time_duration, white_background)
            process.stdin.write(np.ascontiguousarray(image).tobytes())
            progress_callback((shot_frame + 1) / duration_frames, shot_frame + 1)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"ffmpeg failed ({return_code}): {error.strip()}")
    except Exception:
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        process.terminate()
        process.wait(timeout=5)
        raise


class ClientState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dirty = threading.Event()
        self.alive = True
        self.snapshot: CameraSnapshot | None = None
        self.shot_playing = False
        self.final_rendering = False
        self.render_cancel = threading.Event()
        self.dynamic_frame_override: float | None = None


def create_viser_server(host: str, port: int) -> Any:
    """Create a Viser server that serves the bundled 4C4D control client."""
    import viser
    from viser import _viser as viser_impl

    client_root = Path(__file__).resolve().parent / "viewer_client"
    client_index = client_root / "index.html"
    if not client_index.is_file():
        raise RuntimeError(f"Bundled Viser client not found: {client_index}")

    base_server_class = viser_impl.infra.WebsockServer

    class BundledClientWebsockServer(base_server_class):
        def __init__(self, *server_args: Any, **server_kwargs: Any) -> None:
            server_kwargs["http_server_root"] = client_root
            super().__init__(*server_args, **server_kwargs)

    # Viser currently fixes its client path inside ViserServer.__init__. Swap
    # the transport class only while that constructor captures the HTTP root.
    viser_impl.infra.WebsockServer = BundledClientWebsockServer
    try:
        return viser.ViserServer(host=host, port=port)
    finally:
        viser_impl.infra.WebsockServer = base_server_class


def run_server(args: argparse.Namespace, model: Any, pipe: Any, iteration: int, time_duration: tuple[float, float], references: dict[str, ReferenceCamera]) -> None:
    server = create_viser_server(args.host, args.port)
    server.scene.set_up_direction((0.0, 0.0, 1.0))
    initial_reference = references.get(args.initial_camera, next(iter(references.values())))
    initial_wxyz, initial_position = initial_reference.viser_pose()
    server.initial_camera.wxyz = initial_wxyz
    server.initial_camera.position = initial_position

    states: dict[int, ClientState] = {}
    states_lock = threading.Lock()
    render_lock = threading.Lock()

    @server.on_client_connect
    def on_connect(client: Any) -> None:
        state = ClientState()
        with states_lock:
            states[client.client_id] = state

        with client.atomic():
            client.camera.wxyz = initial_wxyz
            client.camera.position = initial_position
            client.camera.fov = initial_reference.fov_y
        aspect_options = {
            "16:9 · UHD": 16.0 / 9.0,
            "2.39:1 · Scope": 2.39,
            "1.85:1 · Flat": 1.85,
            "4:3 · Academy": 4.0 / 3.0,
            "9:16 · Vertical": 9.0 / 16.0,
        }
        initial_shot_aspect = aspect_options["16:9 · UHD"]
        sensor_presets: dict[str, tuple[float, float] | None] = {
            "Super 35 · 4-perf (24.89 × 18.66)": (24.89, 18.66),
            "Super 35 · 3-perf (24.89 × 14.00)": (24.89, 14.00),
            "Full Frame · 16:9 (36.00 × 20.25)": (36.00, 20.25),
            "Full Frame · Stills (36.00 × 24.00)": (36.00, 24.00),
            "APS-C (23.60 × 15.70)": (23.60, 15.70),
            "Micro Four Thirds (17.30 × 13.00)": (17.30, 13.00),
            "Custom gate": None,
        }
        sensor_preset_default = "Super 35 · 4-perf (24.89 × 18.66)"
        sensor_width_default, sensor_height_default = sensor_presets[sensor_preset_default]  # type: ignore[misc]
        initial_focal = float(
            np.clip(
                shot_fov_y_to_focal_length(
                    initial_reference.fov_y,
                    sensor_width_default,
                    sensor_height_default,
                    initial_shot_aspect,
                ),
                MIN_FOCAL_LENGTH_MM,
                MAX_FOCAL_LENGTH_MM,
            )
        )
        initial_shot_fov_y = focal_length_to_shot_fov_y(
            initial_focal, sensor_width_default, sensor_height_default, initial_shot_aspect
        )
        client.camera.fov = initial_shot_fov_y
        initial_horizontal_fov = math.degrees(
            2.0 * math.atan(math.tan(initial_shot_fov_y * 0.5) * initial_shot_aspect)
        )
        shot_duration_default = min(max(args.shot_frames, 2), 600)
        keyframes = [
            ShotKeyframe(0, initial_wxyz.copy(), initial_position.copy(), initial_shot_fov_y),
            ShotKeyframe(
                shot_duration_default - 1,
                initial_wxyz.copy(),
                initial_position.copy(),
                initial_shot_fov_y,
            ),
        ]
        path_handles: list[Any] = []
        gui_guard = False
        last_shot_aspect = initial_shot_aspect
        last_sensor_dimensions = (sensor_width_default, sensor_height_default)

        with client.gui.add_folder("4C4D Playback", expand_by_default=False):
            frame_slider = client.gui.add_slider("Frame", min=0, max=args.frames - 1, step=1, initial_value=args.frame)
            play = client.gui.add_checkbox("Play", initial_value=False)
            fps = client.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=args.fps)
            preview_resolution_mode = client.gui.add_dropdown(
                "Preview resolution", ("Match viewport", "Manual width"), initial_value="Match viewport"
            )
            render_width = client.gui.add_slider(
                "Manual render width", min=320, max=args.max_width, step=32, initial_value=args.width, disabled=True
            )
            timestamp_text = client.gui.add_text("Timestamp", initial_value=f"{frame_to_timestamp(args.frame, args.frames, time_duration):.4f}", disabled=True)
            toggle_playback = client.gui.add_command(
                "Play / pause",
                description="Toggle playback (Space)",
                hotkey="space",
            )
        with client.gui.add_folder("Camera & Lens"):
            with client.gui.add_folder("Camera Transform", expand_by_default=False):
                camera_dropdown = client.gui.add_dropdown("Input camera", tuple(references.keys()), initial_value=initial_reference.name)
                snap_button = client.gui.add_button("Snap to camera")
                camera_position = client.gui.add_vector3("Position XYZ", initial_value=tuple(initial_position), step=0.01)
                initial_euler = rotation_matrix_to_euler_xyz_degrees(quaternion_wxyz_to_matrix(initial_wxyz))
                camera_rotation = client.gui.add_vector3("Rotation XYZ°", initial_value=tuple(initial_euler), step=0.1)
                coordinate_text = client.gui.add_text("Camera coordinates", initial_value="", multiline=True, disabled=True)
            with client.gui.add_folder("Lens & Filmback"):
                sensor_preset = client.gui.add_dropdown("Sensor preset", tuple(sensor_presets), initial_value=sensor_preset_default)
                filmback_lens = client.gui.add_vector3(
                    "Filmback W · H · Focal (mm)",
                    initial_value=(sensor_width_default, sensor_height_default, initial_focal),
                    min=(1.0, 1.0, MIN_FOCAL_LENGTH_MM),
                    max=(100.0, 100.0, MAX_FOCAL_LENGTH_MM),
                    step=0.01,
                    hint="Sensor width, sensor height, and focal length",
                )
                field_of_view = client.gui.add_vector2(
                    "FOV V · H (degrees)",
                    initial_value=(math.degrees(initial_shot_fov_y), initial_horizontal_fov),
                    min=(1.0, 1.0),
                    max=(179.0, 179.0),
                    step=0.1,
                    hint="Vertical and horizontal field of view",
                )
                equivalent_focal = client.gui.add_number(
                    "35mm equivalent (mm)",
                    initial_value=focal_length_to_full_frame_equivalent(
                        initial_focal,
                        sensor_width_default,
                        sensor_height_default,
                        initial_shot_aspect,
                    ),
                    min=focal_length_to_full_frame_equivalent(
                        MIN_FOCAL_LENGTH_MM,
                        sensor_width_default,
                        sensor_height_default,
                        initial_shot_aspect,
                    ),
                    max=focal_length_to_full_frame_equivalent(
                        MAX_FOCAL_LENGTH_MM,
                        sensor_width_default,
                        sensor_height_default,
                        initial_shot_aspect,
                    ),
                    step=0.1,
                    hint="Editable full-frame-equivalent focal length",
                )
        with client.gui.add_folder("Cinematic Shot"):
            shot_name = client.gui.add_text("Shot name", initial_value="shot_001_hero")
            shot_duration = client.gui.add_number("Duration (frames)", initial_value=shot_duration_default, min=2, max=600, step=1)
            shot_frame = client.gui.add_slider("Shot frame", min=0, max=599, step=1, initial_value=0)
            shot_interpolation = client.gui.add_dropdown("Interpolation", ("Smooth ease", "Linear"), initial_value="Smooth ease")
            preview_shot = client.gui.add_checkbox("Preview camera move", initial_value=False)
            loop_shot = client.gui.add_checkbox("Loop shot", initial_value=True)
            aspect_preset = client.gui.add_dropdown("Shot framing", tuple(aspect_options), initial_value="16:9 · UHD")
            match_preview_aspect = client.gui.add_checkbox("Preview shot gate", initial_value=True)
            matte_opacity = client.gui.add_slider("Outside matte", min=0, max=95, step=5, initial_value=50)
            rule_of_thirds = client.gui.add_checkbox("Rule of thirds", initial_value=False)
            action_safe = client.gui.add_checkbox("Action-safe frame", initial_value=False)
            show_camera_path = client.gui.add_checkbox("Show camera path", initial_value=True)
            keyframe_select = client.gui.add_dropdown("Keyframes", ("Frame 0", f"Frame {shot_duration_default - 1}"), initial_value="Frame 0")
            sequencer_data = client.gui.add_text("Sequencer key data", initial_value="{}", disabled=True)
            add_keyframe = client.gui.add_button("Add / update keyframe", color="violet")
            delete_keyframe = client.gui.add_button("Delete selected keyframe")
            export_format = client.gui.add_dropdown("Camera export", ("glTF 2.0", "USD ASCII", "4C4D JSON"), initial_value="glTF 2.0")
            export_camera = client.gui.add_button("Download camera data")
        with client.gui.add_folder("Render & Export", expand_by_default=False):
            final_width = client.gui.add_number("Output width", initial_value=1920, min=320, max=4096, step=2)
            final_resolution = client.gui.add_text("Output resolution", initial_value="1920 × 1080 · 16:9", disabled=True)
            final_fps = client.gui.add_number("Output FPS", initial_value=24, min=1, max=120, step=1)
            final_crf = client.gui.add_slider("H.264 quality (CRF)", min=10, max=35, step=1, initial_value=18)
            render_sidecar = client.gui.add_dropdown("Camera sidecar", ("glTF 2.0", "USD ASCII", "4C4D JSON", "None"), initial_value="glTF 2.0")
            render_mp4 = client.gui.add_button("Render MP4", color="violet")
            cancel_render = client.gui.add_button("Cancel render", disabled=True)
            final_progress = client.gui.add_progress_bar(0.0, visible=False, animated=False, color="violet")
            final_status = client.gui.add_text("Render status", initial_value="Ready", disabled=True)
        with client.gui.add_folder("Performance", expand_by_default=False):
            status = client.gui.add_text("Status", initial_value="Waiting for first render", disabled=True)

        def current_aspect() -> float:
            return aspect_options[str(aspect_preset.value)]

        def update_output_resolution() -> None:
            width = max(2, int(final_width.value))
            width -= width % 2
            height = max(2, int(round(width / current_aspect())))
            height -= height % 2
            final_resolution.value = f"{width} × {height} · {aspect_preset.value}"

        def shot_length() -> int:
            return int(np.clip(int(shot_duration.value), 2, 600))

        def lens_values() -> tuple[float, float, float]:
            width, height, focal = np.asarray(filmback_lens.value, dtype=np.float64)
            return (
                max(float(width), 1.0),
                max(float(height), 1.0),
                float(np.clip(float(focal), MIN_FOCAL_LENGTH_MM, MAX_FOCAL_LENGTH_MM)),
            )

        def lens_fov_bounds(sensor_width_mm: float, sensor_height_mm: float) -> tuple[np.ndarray, np.ndarray]:
            min_vertical = focal_length_to_shot_fov_y(
                MAX_FOCAL_LENGTH_MM, sensor_width_mm, sensor_height_mm, current_aspect()
            )
            max_vertical = focal_length_to_shot_fov_y(
                MIN_FOCAL_LENGTH_MM, sensor_width_mm, sensor_height_mm, current_aspect()
            )
            min_horizontal = 2.0 * math.atan(math.tan(min_vertical * 0.5) * current_aspect())
            max_horizontal = 2.0 * math.atan(math.tan(max_vertical * 0.5) * current_aspect())
            return np.degrees([min_vertical, min_horizontal]), np.degrees(
                [max_vertical, max_horizontal]
            )

        def remap_keyframes_for_sensor(sensor_width_mm: float, sensor_height_mm: float) -> bool:
            nonlocal last_sensor_dimensions
            old_width_mm, old_height_mm = last_sensor_dimensions
            if np.allclose(
                (sensor_width_mm, sensor_height_mm), last_sensor_dimensions, atol=1e-6
            ):
                return False
            for key in keyframes:
                focal_mm = shot_fov_y_to_focal_length(
                    key.fov_y, old_width_mm, old_height_mm, current_aspect()
                )
                key.fov_y = focal_length_to_shot_fov_y(
                    focal_mm, sensor_width_mm, sensor_height_mm, current_aspect()
                )
            last_sensor_dimensions = (sensor_width_mm, sensor_height_mm)
            return True

        def update_coordinate_text() -> None:
            position = np.asarray(client.camera.position, dtype=np.float64)
            quaternion = np.asarray(client.camera.wxyz, dtype=np.float64)
            euler = rotation_matrix_to_euler_xyz_degrees(quaternion_wxyz_to_matrix(quaternion))
            coordinate_text.value = (
                f"Position  X {position[0]:.5f}   Y {position[1]:.5f}   Z {position[2]:.5f}\n"
                f"Rotation  X {euler[0]:.3f}°   Y {euler[1]:.3f}°   Z {euler[2]:.3f}°\n"
                f"Quaternion WXYZ  {quaternion[0]:.7f}, {quaternion[1]:.7f}, {quaternion[2]:.7f}, {quaternion[3]:.7f}"
            )

        def sync_camera_gui() -> None:
            nonlocal gui_guard
            gui_guard = True
            try:
                camera_position.value = tuple(float(value) for value in client.camera.position)
                camera_rotation.value = tuple(
                    float(value)
                    for value in rotation_matrix_to_euler_xyz_degrees(
                        quaternion_wxyz_to_matrix(np.asarray(client.camera.wxyz, dtype=np.float64))
                    )
                )
                sensor_width_mm, sensor_height_mm, _ = lens_values()
                minimum_fov, maximum_fov = lens_fov_bounds(sensor_width_mm, sensor_height_mm)
                field_of_view.min = tuple(float(value) for value in minimum_fov)
                field_of_view.max = tuple(float(value) for value in maximum_fov)
                equivalent_focal.min = focal_length_to_full_frame_equivalent(
                    MIN_FOCAL_LENGTH_MM,
                    sensor_width_mm,
                    sensor_height_mm,
                    current_aspect(),
                )
                equivalent_focal.max = focal_length_to_full_frame_equivalent(
                    MAX_FOCAL_LENGTH_MM,
                    sensor_width_mm,
                    sensor_height_mm,
                    current_aspect(),
                )
                bounded_fov_y = float(
                    np.clip(
                        float(client.camera.fov),
                        math.radians(float(minimum_fov[0])),
                        math.radians(float(maximum_fov[0])),
                    )
                )
                if not math.isclose(bounded_fov_y, float(client.camera.fov)):
                    client.camera.fov = bounded_fov_y
                focal_mm = shot_fov_y_to_focal_length(
                    bounded_fov_y, sensor_width_mm, sensor_height_mm, current_aspect()
                )
                filmback_lens.value = (sensor_width_mm, sensor_height_mm, focal_mm)
                field_of_view.value = (
                    math.degrees(bounded_fov_y),
                    math.degrees(
                        2.0 * math.atan(math.tan(bounded_fov_y * 0.5) * current_aspect())
                    ),
                )
                equivalent_focal.value = focal_length_to_full_frame_equivalent(
                    focal_mm, sensor_width_mm, sensor_height_mm, current_aspect()
                )
                update_coordinate_text()
            finally:
                gui_guard = False

        def refresh_keyframe_gui() -> None:
            ordered = sorted(keyframes, key=lambda item: item.shot_frame)
            labels = tuple(f"Frame {key.shot_frame}" for key in ordered)
            keyframe_select.options = labels
            selected = f"Frame {int(shot_frame.value)}"
            keyframe_select.value = selected if selected in labels else labels[0]
            sequencer_data.value = json.dumps(
                {
                    "keys": [
                        {
                            "frame": key.shot_frame,
                            "focal_mm": round(
                                shot_fov_y_to_focal_length(
                                    key.fov_y, lens_values()[0], lens_values()[1], current_aspect()
                                ),
                                4,
                            ),
                            "fov_degrees": round(math.degrees(key.fov_y), 4),
                        }
                        for key in ordered
                    ]
                },
                separators=(",", ":"),
            )

        def refresh_camera_path() -> None:
            for handle in path_handles:
                handle.remove()
            path_handles.clear()
            if not bool(show_camera_path.value) or not keyframes:
                return
            ordered = sorted(keyframes, key=lambda item: item.shot_frame)
            if len(ordered) > 1:
                sampled = np.asarray(
                    [
                        interpolate_keyframes(
                            ordered,
                            frame,
                            smooth=str(shot_interpolation.value) == "Smooth ease",
                        ).position
                        for frame in range(shot_length())
                    ],
                    dtype=np.float32,
                )
                segments = np.stack((sampled[:-1], sampled[1:]), axis=1)
                path_handles.append(
                    client.scene.add_line_segments(
                        "/cinematic/camera_path",
                        points=segments,
                        line_width=3.0,
                        colors=(155, 124, 255),
                    )
                )
            for index, key in enumerate(ordered):
                path_handles.append(
                    client.scene.add_camera_frustum(
                        f"/cinematic/key_{index:03d}",
                        fov=key.fov_y,
                        aspect=current_aspect(),
                        scale=0.16,
                        line_width=1.5,
                        color=(155, 124, 255) if index else (255, 190, 90),
                        wxyz=key.wxyz,
                        position=key.position,
                    )
                )

        def capture_snapshot() -> CameraSnapshot:
            canvas_width = max(int(client.camera.image_width), 1)
            canvas_height = max(int(client.camera.image_height), 1)
            canvas_aspect = float(client.camera.aspect) if float(client.camera.aspect) > 0 else canvas_width / canvas_height
            shot_aspect = current_aspect()
            preview_framing = bool(match_preview_aspect.value)
            aspect = canvas_aspect
            render_fov_y = float(client.camera.fov)
            if preview_framing:
                render_fov_y = preview_fov_y_for_gate(render_fov_y, canvas_aspect, shot_aspect)
            if str(preview_resolution_mode.value) == "Match viewport":
                preview_width = int(np.clip(canvas_width, 64, args.max_width))
            else:
                preview_width = int(render_width.value)
            return CameraSnapshot(
                wxyz=np.asarray(client.camera.wxyz, dtype=np.float64).copy(),
                position=np.asarray(client.camera.position, dtype=np.float64).copy(),
                fov_y=render_fov_y,
                aspect=aspect,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                frame=(
                    state.dynamic_frame_override
                    if state.dynamic_frame_override is not None
                    else float(frame_slider.value)
                ),
                render_width=preview_width,
                shot_aspect=shot_aspect,
                viewport_aspect=canvas_aspect,
                preview_framing=preview_framing,
            )

        def request_render(_: Any = None) -> None:
            timestamp_text.value = f"{frame_to_timestamp(int(frame_slider.value), args.frames, time_duration):.4f}"
            with state.lock:
                state.snapshot = capture_snapshot()
            state.dirty.set()

        @client.camera.on_update
        def _(_: Any) -> None:
            if not gui_guard:
                state.dynamic_frame_override = None
                sync_camera_gui()
            request_render()

        @frame_slider.on_update
        def _(_: Any) -> None:
            if not gui_guard:
                state.dynamic_frame_override = None
            request_render()
        render_width.on_update(request_render)
        match_preview_aspect.on_update(request_render)
        matte_opacity.on_update(request_render)
        rule_of_thirds.on_update(request_render)
        action_safe.on_update(request_render)

        @preview_resolution_mode.on_update
        def _(_: Any) -> None:
            render_width.disabled = str(preview_resolution_mode.value) == "Match viewport"
            request_render()

        @camera_position.on_update
        def _(_: Any) -> None:
            if gui_guard:
                return
            client.camera.position = np.asarray(camera_position.value, dtype=np.float64)

        @camera_rotation.on_update
        def _(_: Any) -> None:
            if gui_guard:
                return
            client.camera.wxyz = euler_xyz_degrees_to_quaternion(np.asarray(camera_rotation.value, dtype=np.float64))

        @filmback_lens.on_update
        def _(_: Any) -> None:
            nonlocal gui_guard
            if gui_guard:
                return
            following_shot = state.dynamic_frame_override is not None
            sensor_width_mm, sensor_height_mm, focal_mm = lens_values()
            sensor_changed = remap_keyframes_for_sensor(sensor_width_mm, sensor_height_mm)
            preset = sensor_presets[str(sensor_preset.value)]
            if preset is not None and not np.allclose(
                (sensor_width_mm, sensor_height_mm), preset, atol=1e-5
            ):
                gui_guard = True
                try:
                    sensor_preset.value = "Custom gate"
                finally:
                    gui_guard = False
            if sensor_changed and following_shot:
                apply_shot_frame(float(shot_frame.value))
            else:
                client.camera.fov = focal_length_to_shot_fov_y(
                    focal_mm, sensor_width_mm, sensor_height_mm, current_aspect()
                )
            refresh_keyframe_gui()

        @field_of_view.on_update
        def _(_: Any) -> None:
            if gui_guard:
                return
            sensor_width_mm, sensor_height_mm, _ = lens_values()
            minimum_fov, maximum_fov = lens_fov_bounds(sensor_width_mm, sensor_height_mm)
            requested = np.clip(
                np.asarray(field_of_view.value, dtype=np.float64), minimum_fov, maximum_fov
            )
            current_vertical = math.degrees(float(client.camera.fov))
            current_horizontal = math.degrees(
                2.0 * math.atan(math.tan(float(client.camera.fov) * 0.5) * current_aspect())
            )
            if abs(float(requested[1]) - current_horizontal) > abs(float(requested[0]) - current_vertical):
                client.camera.fov = 2.0 * math.atan(
                    math.tan(math.radians(float(requested[1])) * 0.5) / current_aspect()
                )
            else:
                client.camera.fov = math.radians(float(requested[0]))

        @equivalent_focal.on_update
        def _(_: Any) -> None:
            nonlocal gui_guard
            if gui_guard:
                return
            sensor_width_mm, sensor_height_mm, _ = lens_values()
            focal_mm = float(
                np.clip(
                    full_frame_equivalent_to_focal_length(
                        float(equivalent_focal.value),
                        sensor_width_mm,
                        sensor_height_mm,
                        current_aspect(),
                    ),
                    MIN_FOCAL_LENGTH_MM,
                    MAX_FOCAL_LENGTH_MM,
                )
            )
            gui_guard = True
            try:
                filmback_lens.value = (sensor_width_mm, sensor_height_mm, focal_mm)
            finally:
                gui_guard = False
            client.camera.fov = focal_length_to_shot_fov_y(
                focal_mm, sensor_width_mm, sensor_height_mm, current_aspect()
            )

        @sensor_preset.on_update
        def _(_: Any) -> None:
            nonlocal gui_guard
            preset = sensor_presets[str(sensor_preset.value)]
            if preset is None or gui_guard:
                return
            following_shot = state.dynamic_frame_override is not None
            _, _, current_focal = lens_values()
            remap_keyframes_for_sensor(preset[0], preset[1])
            gui_guard = True
            try:
                filmback_lens.value = (preset[0], preset[1], current_focal)
            finally:
                gui_guard = False
            if following_shot:
                apply_shot_frame(float(shot_frame.value))
            else:
                client.camera.fov = focal_length_to_shot_fov_y(
                    current_focal, preset[0], preset[1], current_aspect()
                )
            refresh_keyframe_gui()

        @toggle_playback.on_trigger
        def _(_: Any) -> None:
            play.value = not bool(play.value)

        @snap_button.on_click
        def _(_: Any) -> None:
            reference = references[str(camera_dropdown.value)]
            wxyz, position = reference.viser_pose()
            with client.atomic():
                client.camera.wxyz = wxyz
                client.camera.position = position
                client.camera.fov = reference.fov_y
            request_render()

        def apply_shot_frame(target_frame: float) -> None:
            nonlocal gui_guard
            key = interpolate_keyframes(keyframes, target_frame, str(shot_interpolation.value) == "Smooth ease")
            dynamic_frame = scene_frame_for_shot(target_frame, float(final_fps.value), args.frames, time_duration)
            state.dynamic_frame_override = dynamic_frame
            gui_guard = True
            try:
                with client.atomic():
                    client.camera.wxyz = key.wxyz
                    client.camera.position = key.position
                    client.camera.fov = key.fov_y
                    frame_slider.value = int(round(dynamic_frame))
                sync_camera_gui()
            finally:
                gui_guard = False
            request_render()

        @shot_frame.on_update
        def _(_: Any) -> None:
            clipped = int(np.clip(int(shot_frame.value), 0, shot_length() - 1))
            if clipped != int(shot_frame.value):
                shot_frame.value = clipped
                apply_shot_frame(clipped)
                return
            apply_shot_frame(clipped)

        @shot_duration.on_update
        def _(_: Any) -> None:
            last_frame = shot_length() - 1
            for key in keyframes:
                key.shot_frame = min(key.shot_frame, last_frame)
            deduplicated: dict[int, ShotKeyframe] = {key.shot_frame: key for key in keyframes}
            keyframes[:] = sorted(deduplicated.values(), key=lambda key: key.shot_frame)
            if int(shot_frame.value) >= shot_length():
                shot_frame.value = last_frame
            refresh_keyframe_gui()
            refresh_camera_path()
            apply_shot_frame(int(shot_frame.value))

        @add_keyframe.on_click
        def _(_: Any) -> None:
            target = int(np.clip(int(shot_frame.value), 0, shot_length() - 1))
            captured = ShotKeyframe(
                shot_frame=target,
                wxyz=np.asarray(client.camera.wxyz, dtype=np.float64).copy(),
                position=np.asarray(client.camera.position, dtype=np.float64).copy(),
                fov_y=float(client.camera.fov),
            )
            existing = next((index for index, key in enumerate(keyframes) if key.shot_frame == target), None)
            if existing is None:
                keyframes.append(captured)
            else:
                keyframes[existing] = captured
            keyframes.sort(key=lambda key: key.shot_frame)
            refresh_keyframe_gui()
            refresh_camera_path()
            final_status.value = f"Keyed camera frame {target}"

        @keyframe_select.on_update
        def _(_: Any) -> None:
            try:
                selected_frame = int(str(keyframe_select.value).split()[-1])
            except (ValueError, IndexError):
                return
            shot_frame.value = selected_frame
            apply_shot_frame(selected_frame)

        @delete_keyframe.on_click
        def _(_: Any) -> None:
            if len(keyframes) <= 1:
                final_status.value = "A shot must keep at least one keyframe"
                return
            selected_frame = int(shot_frame.value)
            if not any(key.shot_frame == selected_frame for key in keyframes):
                final_status.value = f"No camera keyframe at frame {selected_frame}"
                return
            keyframes[:] = [key for key in keyframes if key.shot_frame != selected_frame]
            refresh_keyframe_gui()
            refresh_camera_path()
            final_status.value = f"Deleted camera keyframe {selected_frame}"

        show_camera_path.on_update(lambda _: refresh_camera_path())

        @shot_interpolation.on_update
        def _(_: Any) -> None:
            refresh_camera_path()
            if state.dynamic_frame_override is not None:
                apply_shot_frame(float(shot_frame.value))

        @final_fps.on_update
        def _(_: Any) -> None:
            if state.dynamic_frame_override is not None:
                apply_shot_frame(float(shot_frame.value))

        @aspect_preset.on_update
        def _(_: Any) -> None:
            nonlocal gui_guard, last_shot_aspect
            following_shot = state.dynamic_frame_override is not None
            new_aspect = current_aspect()
            sensor_width_mm, sensor_height_mm, _ = lens_values()
            current_focal = shot_fov_y_to_focal_length(
                float(client.camera.fov), sensor_width_mm, sensor_height_mm, last_shot_aspect
            )
            for key in keyframes:
                focal_mm = shot_fov_y_to_focal_length(
                    key.fov_y, sensor_width_mm, sensor_height_mm, last_shot_aspect
                )
                key.fov_y = focal_length_to_shot_fov_y(
                    focal_mm, sensor_width_mm, sensor_height_mm, new_aspect
                )
            last_shot_aspect = new_aspect
            if following_shot:
                apply_shot_frame(float(shot_frame.value))
            else:
                gui_guard = True
                try:
                    client.camera.fov = focal_length_to_shot_fov_y(
                        current_focal, sensor_width_mm, sensor_height_mm, new_aspect
                    )
                finally:
                    gui_guard = False
                sync_camera_gui()
            refresh_keyframe_gui()
            refresh_camera_path()
            update_output_resolution()
            request_render()

        final_width.on_update(lambda _: update_output_resolution())

        def camera_export(
            selected_format: str | None = None,
            source_keys: list[ShotKeyframe] | None = None,
            export_fps_value: int | None = None,
            export_sensor_width: float | None = None,
            export_sensor_height: float | None = None,
            export_shot_aspect: float | None = None,
            export_duration_frames: int | None = None,
            export_interpolation: str | None = None,
            export_name: str | None = None,
        ) -> tuple[str, bytes]:
            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in (export_name if export_name is not None else str(shot_name.value))
            ).strip("_") or "camera_shot"
            selected_format = selected_format or str(export_format.value)
            source_keys = source_keys or keyframes
            export_fps_value = export_fps_value or int(final_fps.value)
            sensor_width_mm, sensor_height_mm, _ = lens_values()
            export_sensor_width = export_sensor_width or sensor_width_mm
            export_sensor_height = export_sensor_height or sensor_height_mm
            export_shot_aspect = export_shot_aspect or current_aspect()
            export_duration_frames = export_duration_frames or shot_length()
            smooth = (export_interpolation or str(shot_interpolation.value)) == "Smooth ease"
            source_keys = export_keyframes(
                source_keys,
                export_duration_frames,
                smooth,
                bake_every_frame=selected_format == "USD ASCII",
            )
            if selected_format == "USD ASCII":
                return f"{safe_name}.usda", shot_to_usda_bytes(
                    safe_name,
                    export_fps_value,
                    source_keys,
                    export_sensor_width,
                    export_sensor_height,
                    export_shot_aspect,
                    export_duration_frames,
                )
            if selected_format == "4C4D JSON":
                return f"{safe_name}.camera.json", shot_to_json_bytes(
                    safe_name,
                    export_fps_value,
                    source_keys,
                    export_sensor_width,
                    export_sensor_height,
                    export_shot_aspect,
                    export_duration_frames,
                )
            return f"{safe_name}.gltf", shot_to_gltf_bytes(
                safe_name,
                export_fps_value,
                source_keys,
                export_sensor_width,
                export_sensor_height,
                export_shot_aspect,
                export_duration_frames,
            )

        @export_camera.on_click
        def _(_: Any) -> None:
            filename, content = camera_export()
            client.send_file_download(filename, content, save_immediately=True)
            final_status.value = f"Downloaded {filename}"

        @render_mp4.on_click
        def _(_: Any) -> None:
            if state.final_rendering:
                return
            if len(keyframes) < 2:
                final_status.value = "Add at least two camera keyframes before rendering"
                return
            state.final_rendering = True
            state.render_cancel.clear()
            render_mp4.disabled = True
            cancel_render.disabled = False
            final_progress.visible = True
            final_progress.value = 0.0
            final_status.value = "Preparing MP4 render…"
            copied_keys = [
                ShotKeyframe(key.shot_frame, key.wxyz.copy(), key.position.copy(), key.fov_y)
                for key in keyframes
            ]
            render_settings = {
                "duration": shot_length(),
                "fps": int(final_fps.value),
                "width": int(final_width.value),
                "aspect": current_aspect(),
                "interpolation": str(shot_interpolation.value),
                "crf": int(final_crf.value),
                "sidecar": str(render_sidecar.value),
                "sensor_width": lens_values()[0],
                "sensor_height": lens_values()[1],
                "shot_aspect": current_aspect(),
                "shot_name": str(shot_name.value),
            }

            def final_render_worker() -> None:
                safe_name = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in str(render_settings["shot_name"])
                ).strip("_") or "camera_shot"
                try:
                    with tempfile.TemporaryDirectory(prefix="4c4d-shot-") as temp_dir:
                        output_path = Path(temp_dir) / f"{safe_name}.mp4"

                        def update_progress(progress: float, completed: int) -> None:
                            final_progress.value = progress * 100.0
                            final_status.value = f"Rendering {completed} / {render_settings['duration']} frames"

                        with render_lock:
                            render_shot_mp4(
                                model=model,
                                pipe=pipe,
                                keyframes=copied_keys,
                                duration_frames=int(render_settings["duration"]),
                                fps=int(render_settings["fps"]),
                                width=int(render_settings["width"]),
                                aspect=float(render_settings["aspect"]),
                                interpolation=str(render_settings["interpolation"]),
                                num_scene_frames=args.frames,
                                time_duration=time_duration,
                                white_background=args.white_background,
                                crf=int(render_settings["crf"]),
                                output_path=output_path,
                                progress_callback=update_progress,
                                cancelled=state.render_cancel,
                            )
                        output_size = output_path.stat().st_size
                        max_download_size = 256 * 1024 * 1024
                        if output_size > max_download_size:
                            raise RuntimeError(
                                f"MP4 is {output_size / (1024 * 1024):.1f} MiB; "
                                "the browser download limit is 256 MiB. Lower output width or raise CRF."
                            )
                        client.send_file_download(output_path.name, output_path.read_bytes(), save_immediately=True)
                        if str(render_settings["sidecar"]) != "None":
                            sidecar_name, sidecar_content = camera_export(
                                selected_format=str(render_settings["sidecar"]),
                                source_keys=copied_keys,
                                export_fps_value=int(render_settings["fps"]),
                                export_sensor_width=float(render_settings["sensor_width"]),
                                export_sensor_height=float(render_settings["sensor_height"]),
                                export_shot_aspect=float(render_settings["shot_aspect"]),
                                export_duration_frames=int(render_settings["duration"]),
                                export_interpolation=str(render_settings["interpolation"]),
                                export_name=str(render_settings["shot_name"]),
                            )
                            client.send_file_download(sidecar_name, sidecar_content, save_immediately=True)
                        final_status.value = f"Complete · {output_path.name}"
                except InterruptedError:
                    final_status.value = "Render cancelled"
                except Exception as exc:
                    final_status.value = f"Render failed: {exc}"
                    print(f"Final render failed for client {client.client_id}: {exc}", flush=True)
                finally:
                    state.final_rendering = False
                    render_mp4.disabled = False
                    cancel_render.disabled = True

            threading.Thread(target=final_render_worker, daemon=True, name=f"4c4d-final-{client.client_id}").start()

        @cancel_render.on_click
        def _(_: Any) -> None:
            if state.final_rendering:
                state.render_cancel.set()
                final_status.value = "Cancelling after current frame…"

        def render_worker() -> None:
            while state.alive:
                state.dirty.wait(timeout=0.5)
                if not state.alive:
                    return
                if not state.dirty.is_set():
                    continue
                state.dirty.clear()
                with state.lock:
                    snapshot = state.snapshot
                if snapshot is None:
                    continue
                try:
                    status.value = f"Rendering frame {snapshot.frame}..."
                    # The custom CUDA rasterizer shares one model and stream. Serialize
                    # clients so two browser tabs cannot race through the extension.
                    with render_lock:
                        image, elapsed = render_image(model, pipe, snapshot, args.frames, time_duration, args.white_background)
                    if snapshot.preview_framing:
                        image = compose_framed_preview(
                            image,
                            viewport_aspect=snapshot.viewport_aspect or snapshot.aspect,
                            shot_aspect=snapshot.shot_aspect or snapshot.aspect,
                            matte_opacity=float(matte_opacity.value) / 100.0,
                            rule_of_thirds=bool(rule_of_thirds.value),
                            action_safe=bool(action_safe.value),
                        )
                    client.scene.set_background_image(image, format="jpeg", jpeg_quality=args.jpeg_quality)
                    status.value = f"{image.shape[1]}x{image.shape[0]} in {elapsed * 1000:.0f} ms ({1.0 / elapsed:.1f} FPS)"
                except Exception as exc:
                    status.value = f"Render failed: {exc}"
                    print(f"Render failed for client {client.client_id}: {exc}", flush=True)

        def playback_worker() -> None:
            while state.alive:
                if bool(preview_shot.value):
                    state.shot_playing = True
                    next_frame = int(shot_frame.value) + 1
                    if next_frame >= shot_length():
                        if bool(loop_shot.value):
                            next_frame = 0
                        else:
                            preview_shot.value = False
                            state.shot_playing = False
                            continue
                    shot_frame.value = next_frame
                    apply_shot_frame(next_frame)
                    time.sleep(1.0 / max(int(final_fps.value), 1))
                elif bool(play.value):
                    state.shot_playing = False
                    state.dynamic_frame_override = None
                    frame_slider.value = (int(frame_slider.value) + 1) % args.frames
                    request_render()
                    time.sleep(1.0 / max(int(fps.value), 1))
                else:
                    state.shot_playing = False
                    time.sleep(0.05)

        threading.Thread(target=render_worker, daemon=True, name=f"4c4d-render-{client.client_id}").start()
        threading.Thread(target=playback_worker, daemon=True, name=f"4c4d-playback-{client.client_id}").start()
        sync_camera_gui()
        refresh_keyframe_gui()
        refresh_camera_path()
        update_output_resolution()
        request_render()

    @server.on_client_disconnect
    def on_disconnect(client: Any) -> None:
        with states_lock:
            state = states.pop(client.client_id, None)
        if state is not None:
            state.alive = False
            state.render_cancel.set()
            state.dirty.set()

    print(f"4C4D iteration {iteration:,} viewer ready at http://localhost:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer.", flush=True)


def self_test() -> None:
    rotations = [
        np.eye(3),
        quaternion_wxyz_to_matrix(np.array([0.5, 0.5, 0.5, 0.5])),
        quaternion_wxyz_to_matrix(np.array([0.8, -0.2, 0.3, 0.45])),
    ]
    for rotation in rotations:
        recovered = quaternion_wxyz_to_matrix(matrix_to_quaternion_wxyz(rotation))
        np.testing.assert_allclose(rotation, recovered, atol=1e-7)
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = rotations[-1]
    c2w_cv[:3, 3] = [1.0, 2.0, 3.0]
    wxyz, position = opencv_c2w_to_viser(c2w_cv)
    np.testing.assert_allclose(opengl_c2w_to_opencv_c2w(wxyz, position), c2w_cv, atol=1e-7)
    assert frame_to_timestamp(0, 300, (0.0, 10.0)) == 0.0
    assert math.isclose(frame_to_timestamp(299, 300, (0.0, 10.0)), 299.0 / 30.0)
    assert math.isclose(
        full_frame_equivalent_to_focal_length(50.0, 24.89, 18.66, 16.0 / 9.0),
        34.5694444444,
    )
    assert math.isclose(
        focal_length_to_full_frame_equivalent(
            34.5694444444, 24.89, 18.66, 16.0 / 9.0
        ),
        50.0,
    )
    portrait_equivalent = focal_length_to_full_frame_equivalent(
        50.0, 24.89, 18.66, 9.0 / 16.0
    )
    assert math.isclose(portrait_equivalent, 50.0 * 13.5 / (18.66 * 9.0 / 16.0))
    assert math.isclose(
        full_frame_equivalent_to_focal_length(
            portrait_equivalent, 24.89, 18.66, 9.0 / 16.0
        ),
        50.0,
    )
    gate_width, gate_height = shot_gate_dimensions(24.89, 18.66, 16.0 / 9.0)
    assert math.isclose(gate_width, 24.89)
    assert math.isclose(gate_height, 24.89 / (16.0 / 9.0))
    test_shot_fov = focal_length_to_shot_fov_y(50.0, 24.89, 18.66, 16.0 / 9.0)
    assert math.isclose(shot_fov_y_to_focal_length(test_shot_fov, 24.89, 18.66, 16.0 / 9.0), 50.0)
    for focal_limit in (MIN_FOCAL_LENGTH_MM, MAX_FOCAL_LENGTH_MM):
        bounded_fov = focal_length_to_shot_fov_y(
            focal_limit, 24.89, 18.66, 16.0 / 9.0
        )
        assert math.isclose(
            shot_fov_y_to_focal_length(bounded_fov, 24.89, 18.66, 16.0 / 9.0),
            focal_limit,
        )
    assert math.isclose(preview_fov_y_for_gate(test_shot_fov, 2.0, 16.0 / 9.0), test_shot_fov)
    assert preview_fov_y_for_gate(test_shot_fov, 4.0 / 3.0, 16.0 / 9.0) > test_shot_fov
    euler = np.array([12.0, -24.0, 5.0])
    euler_round_trip = rotation_matrix_to_euler_xyz_degrees(
        quaternion_wxyz_to_matrix(euler_xyz_degrees_to_quaternion(euler))
    )
    np.testing.assert_allclose(euler, euler_round_trip, atol=1e-7)
    test_keys = [
        ShotKeyframe(0, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), math.radians(45.0)),
        ShotKeyframe(24, np.array([1.0, 0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.0]), math.radians(60.0)),
    ]
    midpoint = interpolate_keyframes(test_keys, 12, smooth=False)
    np.testing.assert_allclose(midpoint.position, [1.0, 0.5, 0.0])
    assert scene_frame_for_shot(12, 24, 300, (0.0, 10.0)) == 15.0
    smooth_export = export_keyframes(test_keys, 30, smooth=True)
    assert len(smooth_export) == 30 and smooth_export[-1].shot_frame == 29
    linear_export = export_keyframes(test_keys, 30, smooth=False)
    assert linear_export[0].shot_frame == 0 and linear_export[-1].shot_frame == 29
    linear_baked_export = export_keyframes(
        test_keys, 30, smooth=False, bake_every_frame=True
    )
    assert len(linear_baked_export) == 30 and linear_baked_export[12].shot_frame == 12
    json_export = json.loads(
        shot_to_json_bytes("test", 24, linear_export, 24.89, 18.66, 16.0 / 9.0, 30)
    )
    assert json_export["sensor_width_mm"] == 24.89
    assert math.isclose(json_export["shot_aspect_ratio"], 16.0 / 9.0)
    assert json_export["duration_frames"] == 30
    assert "horizontal_fov_degrees" in json_export["keyframes"][0]
    assert "scene_frame" not in json_export["keyframes"][0]
    assert "focus_distance" not in json_export["keyframes"][0]
    assert "aperture" not in json_export["keyframes"][0]
    gltf = json.loads(shot_to_gltf_bytes("test", 24, smooth_export, 24.89, 18.66, 16.0 / 9.0, 30))
    assert gltf["asset"]["version"] == "2.0"
    assert math.isclose(gltf["cameras"][0]["perspective"]["aspectRatio"], 16.0 / 9.0)
    assert gltf["nodes"][0]["children"] == [1]
    assert gltf["animations"][0]["channels"][0]["target"]["node"] == 1
    assert (
        gltf["animations"][0]["channels"][2]["target"]["extensions"]["KHR_animation_pointer"]["pointer"]
        == "/cameras/0/perspective/yfov"
    )
    assert gltf["extras"]["duration_frames"] == 30
    assert "scene_frame" not in gltf["extras"]
    assert "focus_distance" not in gltf["extras"]
    assert "aperture" not in gltf["extras"]
    usda = shot_to_usda_bytes(
        "001 hé", 24, linear_baked_export, 24.89, 18.66, 16.0 / 9.0, 30
    )
    assert b"focusDistance" not in usda
    assert b"fStop" not in usda
    assert b"horizontalAperture = 24.89" in usda
    assert b'defaultPrim = "_001_h_"' in usda
    assert b"endTimeCode = 29" in usda
    assert f"verticalAperture = {gate_height}".encode() in usda
    assert b"double3 xformOp:translate.timeSamples" in usda
    assert b"quatd xformOp:orient.timeSamples" in usda
    assert b"matrix4d" not in usda
    assert b"            12:" in usda
    preview_wide = np.full((180, 320, 3), 200, dtype=np.uint8)
    framed = compose_framed_preview(preview_wide, 16.0 / 9.0, 4.0 / 3.0, 0.75, True, True)
    assert framed.shape == preview_wide.shape
    assert framed[:, 0].mean() < framed[:, 160].mean()
    preview_tall = np.full((240, 320, 3), 200, dtype=np.uint8)
    letterboxed = compose_framed_preview(preview_tall, 4.0 / 3.0, 16.0 / 9.0, 0.85, False, False)
    assert 0.0 < letterboxed[0].mean() < letterboxed[120].mean()
    print("Viewer math self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="4C4D YAML configuration")
    parser.add_argument("--checkpoint", type=Path, help="4C4D .pth checkpoint")
    parser.add_argument("--cameras-json", type=Path, default=None)
    parser.add_argument("--training-views", default="1,10,13,20")
    parser.add_argument("--initial-camera", default="cam10")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--frame", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--shot-frames", type=int, default=120, help="Default cinematic shot duration")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.config is None or args.checkpoint is None:
        raise SystemExit("--config and --checkpoint are required unless --self-test is used")
    if not 0 <= args.frame < args.frames:
        raise SystemExit(f"--frame must be between 0 and {args.frames - 1}")
    args.cameras_json = args.cameras_json or args.checkpoint.parent / "cameras.json"
    training_views = [int(value) for value in args.training_views.split(",") if value.strip()]
    references = load_reference_cameras(args.cameras_json, training_views)
    model, pipe, iteration, time_duration = load_model(args.config, args.checkpoint)
    run_server(args, model, pipe, iteration, time_duration, references)


if __name__ == "__main__":
    main()
