#!/usr/bin/env python3
"""Check whether RGB videos share a usable presentation-time timeline.

This validates container timestamps and cadence without reading recorded depth.
It cannot prove hardware shutter synchronization; use a flash, timecode display,
or cross-camera motion correlation when sub-frame synchronization matters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_frame_indices(value: str) -> list[int]:
    frames = sorted({int(item) for item in value.split(",") if item.strip()})
    if not frames or frames[0] < 0:
        raise argparse.ArgumentTypeError("frames must be non-negative comma-separated integers")
    return frames


def inspect_video(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=start_time,duration,time_base,avg_frame_rate,nb_frames:"
        "frame=best_effort_timestamp_time",
        "-of", "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=180
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe executable not found: {ffprobe}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or f"exit status {exc.returncode}"
        raise RuntimeError(f"ffprobe failed for {path}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out for {path}") from exc

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in payload.get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    ]
    if not timestamps:
        raise RuntimeError(f"No frame presentation timestamps found in {path}")
    return {"path": str(path), "stream": streams[0], "timestamps": timestamps}


def analyze_timelines(
    videos: list[dict[str, Any]],
    sample_frames: list[int],
    *,
    max_pts_delta_ms: float = 1.0,
    max_cadence_error_ms: float = 0.2,
) -> dict[str, Any]:
    if len(videos) < 2:
        raise ValueError("At least two RGB videos are required")

    required_frame = max(sample_frames)
    failures, warnings, reports = [], [], []
    sampled_by_frame: dict[int, list[float]] = {frame: [] for frame in sample_frames}
    expected_step = None

    for video in videos:
        timestamps = video["timestamps"]
        if len(timestamps) <= required_frame:
            failures.append(
                f"{video['path']} has {len(timestamps)} timestamped frames; "
                f"frame {required_frame} is required"
            )
            continue
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        median_step = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
        if expected_step is None:
            expected_step = median_step
        max_cadence_error = max(
            (abs(delta - median_step) for delta in deltas), default=0.0
        )
        samples = {str(frame): timestamps[frame] for frame in sample_frames}
        for frame in sample_frames:
            sampled_by_frame[frame].append(timestamps[frame])
        reports.append({
            "path": video["path"],
            "frame_count": len(timestamps),
            "start_time_s": timestamps[0],
            "end_time_s": timestamps[-1],
            "median_frame_duration_ms": median_step * 1000.0,
            "max_cadence_error_ms": max_cadence_error * 1000.0,
            "sample_pts_s": samples,
            "stream": video.get("stream", {}),
        })
        if max_cadence_error * 1000.0 > max_cadence_error_ms:
            failures.append(
                f"{video['path']} cadence error exceeds {max_cadence_error_ms:.3g}ms"
            )

    sample_deltas = {}
    for frame, timestamps in sampled_by_frame.items():
        if len(timestamps) != len(videos):
            continue
        delta_ms = (max(timestamps) - min(timestamps)) * 1000.0
        sample_deltas[str(frame)] = delta_ms
        if delta_ms > max_pts_delta_ms:
            failures.append(
                f"frame {frame} presentation timestamps differ by {delta_ms:.3f}ms"
            )

    frame_counts = {len(video["timestamps"]) for video in videos}
    if len(frame_counts) > 1:
        warnings.append(
            "RGB videos have different total frame counts; the requested sample range is valid"
        )

    return {
        "passed": not failures,
        "limits": {
            "max_pts_delta_ms": max_pts_delta_ms,
            "max_cadence_error_ms": max_cadence_error_ms,
        },
        "sample_frames": sample_frames,
        "sample_pts_delta_ms": sample_deltas,
        "failures": failures,
        "warnings": warnings,
        "videos": reports,
        "limitation": (
            "Matching container PTS does not prove hardware shutter synchronization; "
            "validate image content when sub-frame accuracy matters."
        ),
    }


def videos_from_manifest(path: Path) -> list[Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    videos = [Path(camera["sourceColor"]) for camera in manifest.get("cameras", [])]
    if not videos:
        raise ValueError(f"No cameras[].sourceColor entries found in {path}")
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--videos", type=Path, nargs="+")
    parser.add_argument("--frames", type=parse_frame_indices, default=[0, 300])
    parser.add_argument("--max-pts-delta-ms", type=float, default=1.0)
    parser.add_argument("--max-cadence-error-ms", type=float, default=0.2)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    args = parser.parse_args()

    paths = videos_from_manifest(args.manifest) if args.manifest else args.videos
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing RGB videos:\n  " + "\n  ".join(missing))
    report = analyze_timelines(
        [inspect_video(path, args.ffprobe) for path in paths],
        args.frames,
        max_pts_delta_ms=args.max_pts_delta_ms,
        max_cadence_error_ms=args.max_cadence_error_ms,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_quality_gate and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
