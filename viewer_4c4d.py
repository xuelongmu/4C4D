"""Interactive browser viewer for trained 4C4D checkpoints.

The browser UI is provided by Viser, while all pixels are produced by the
repository's native 4D CUDA rasterizer.  This keeps the temporal Gaussian
conditioning and temporal spherical harmonics identical to render.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


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
    frame: int
    render_width: int


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


class ClientState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dirty = threading.Event()
        self.alive = True
        self.snapshot: CameraSnapshot | None = None


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
        with client.gui.add_folder("4C4D Playback"):
            frame_slider = client.gui.add_slider("Frame", min=0, max=args.frames - 1, step=1, initial_value=args.frame)
            play = client.gui.add_checkbox("Play", initial_value=False)
            fps = client.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=args.fps)
            render_width = client.gui.add_slider("Render width", min=320, max=args.max_width, step=32, initial_value=args.width)
            timestamp_text = client.gui.add_text("Timestamp", initial_value=f"{frame_to_timestamp(args.frame, args.frames, time_duration):.4f}", disabled=True)
            toggle_playback = client.gui.add_command(
                "Play / pause",
                description="Toggle playback (Space)",
                hotkey="space",
            )
        with client.gui.add_folder("Camera"):
            camera_dropdown = client.gui.add_dropdown("Input camera", tuple(references.keys()), initial_value=initial_reference.name)
            snap_button = client.gui.add_button("Snap to camera")
        with client.gui.add_folder("Performance"):
            status = client.gui.add_text("Status", initial_value="Waiting for first render", disabled=True)

        def capture_snapshot() -> CameraSnapshot:
            canvas_width = max(int(client.camera.image_width), 1)
            canvas_height = max(int(client.camera.image_height), 1)
            aspect = float(client.camera.aspect) if float(client.camera.aspect) > 0 else canvas_width / canvas_height
            return CameraSnapshot(
                wxyz=np.asarray(client.camera.wxyz, dtype=np.float64).copy(),
                position=np.asarray(client.camera.position, dtype=np.float64).copy(),
                fov_y=float(client.camera.fov),
                aspect=aspect,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                frame=int(frame_slider.value),
                render_width=int(render_width.value),
            )

        def request_render(_: Any = None) -> None:
            timestamp_text.value = f"{frame_to_timestamp(int(frame_slider.value), args.frames, time_duration):.4f}"
            with state.lock:
                state.snapshot = capture_snapshot()
            state.dirty.set()

        @client.camera.on_update
        def _(_: Any) -> None:
            request_render()

        frame_slider.on_update(request_render)
        render_width.on_update(request_render)

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
                    client.scene.set_background_image(image, format="jpeg", jpeg_quality=args.jpeg_quality)
                    status.value = f"{image.shape[1]}x{image.shape[0]} in {elapsed * 1000:.0f} ms ({1.0 / elapsed:.1f} FPS)"
                except Exception as exc:
                    status.value = f"Render failed: {exc}"
                    print(f"Render failed for client {client.client_id}: {exc}", flush=True)

        def playback_worker() -> None:
            while state.alive:
                if bool(play.value):
                    frame_slider.value = (int(frame_slider.value) + 1) % args.frames
                    request_render()
                    time.sleep(1.0 / max(int(fps.value), 1))
                else:
                    time.sleep(0.05)

        threading.Thread(target=render_worker, daemon=True, name=f"4c4d-render-{client.client_id}").start()
        threading.Thread(target=playback_worker, daemon=True, name=f"4c4d-playback-{client.client_id}").start()
        request_render()

    @server.on_client_disconnect
    def on_disconnect(client: Any) -> None:
        with states_lock:
            state = states.pop(client.client_id, None)
        if state is not None:
            state.alive = False
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
