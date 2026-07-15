"""Record labeled webcam clips and per-frame timestamps for detector evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

from motioncon.config import Settings
from motioncon.vision.camera import Camera

GESTURE_LABELS = (
    "raise_settle_flick_up",
    "raise_settle_flick_down",
    "raise_settle_swipe_left",
)
DISTRACTOR_LABELS = (
    "typing",
    "head_turn",
    "lean_in_out",
    "raise_and_lower_only",
)
LABELS = (*GESTURE_LABELS, *DISTRACTOR_LABELS)
_WINDOW = "motioncon clip recorder"


def record_clip(
    *,
    label: str,
    duration: float,
    camera_index: int,
    output_root: Path,
    settings: Settings,
) -> Path:
    """Capture one labeled MJPEG clip plus a JSON timestamp sidecar."""
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    video_path = output_dir / f"{clip_id}.avi"
    metadata_path = video_path.with_suffix(".timestamps.json")

    frame_times: list[dict[str, float]] = []
    started_wall = time.time()
    started_mono = time.perf_counter()
    writer: cv2.VideoWriter | None = None

    try:
        with Camera(
            index=camera_index,
            width=settings.frame_width,
            height=settings.frame_height,
            mirror=settings.mirror,
        ) as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    break
                elapsed = time.perf_counter() - started_mono
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        30.0,
                        (width, height),
                    )
                    if not writer.isOpened():
                        msg = f"could not create video at {video_path}"
                        raise RuntimeError(msg)

                writer.write(frame)
                frame_times.append({"elapsed_s": elapsed, "wall_time_s": time.time()})

                preview = frame.copy()
                top = int(settings.gesture_band[0] * preview.shape[0])
                bottom = int(settings.gesture_band[1] * preview.shape[0])
                cv2.line(preview, (0, top), (preview.shape[1], top), (120, 120, 120), 1)
                cv2.line(preview, (0, bottom), (preview.shape[1], bottom), (120, 120, 120), 1)
                cv2.putText(
                    preview,
                    f"{label}  {elapsed:.1f}/{duration:.1f}s",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(_WINDOW, preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or elapsed >= duration:
                    break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if not frame_times:
        msg = "camera produced no frames"
        raise RuntimeError(msg)

    metadata = {
        "label": label,
        "camera_index": camera_index,
        "mirrored": settings.mirror,
        "started_wall_time_s": started_wall,
        "duration_s": frame_times[-1]["elapsed_s"],
        "frame_width": settings.frame_width,
        "frame_height": settings.frame_height,
        "frame_timestamps": frame_times,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=LABELS)
    parser.add_argument("--duration", type=float, default=3.0, help="clip length, 2 to 5 seconds")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/clips"))
    args = parser.parse_args()
    if not 2.0 <= args.duration <= 5.0:
        parser.error("--duration must be between 2 and 5 seconds")

    path = record_clip(
        label=args.label,
        duration=args.duration,
        camera_index=args.camera,
        output_root=args.output,
        settings=Settings(camera_index=args.camera),
    )
    print(path)


if __name__ == "__main__":
    main()
