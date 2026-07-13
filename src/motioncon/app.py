"""Application loop (I/O edge): wires the pure modules together.

capture -> difference -> motion -> gestures -> menu update -> render -> HUD ->
keys. This module contains no core algorithms, only orchestration, HUD text,
and key handling.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import numpy.typing as npt

from motioncon.config import DitherMode, Event, Settings
from motioncon.control.gestures import GestureRecognizer, PushSelectDetector
from motioncon.render.dither import dither
from motioncon.render.dots import downsample, render_dots
from motioncon.telemetry.logger import JsonlLogger
from motioncon.ui.menu import Menu, MenuItem
from motioncon.vision.camera import Camera
from motioncon.vision.difference import TrailsAccumulator, boost, frame_difference, to_gray
from motioncon.vision.motion import MotionAnalyzer, MotionState

_WINDOW = "motion-console"

_DEMO_MENU = (
    MenuItem("Play", action="play"),
    MenuItem(
        "Gallery",
        children=(
            MenuItem("Photos", action="gallery.photos"),
            MenuItem("Videos", action="gallery.videos"),
        ),
    ),
    MenuItem(
        "Settings",
        children=(
            MenuItem("Sensitivity", action="settings.sensitivity"),
            MenuItem("Colors", action="settings.colors"),
        ),
    ),
    MenuItem("About", action="about"),
)


def _draw_hud(
    canvas: npt.NDArray[np.uint8],
    menu: Menu,
    state: MotionState,
    cursor: tuple[float, float] | None,
    last_event: Event | None,
    fps: float,
) -> npt.NDArray[np.uint8]:
    """Overlay menu, cursor marker, and status text onto a BGR canvas."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, item in enumerate(menu.items):
        selected = i == menu.selected_index
        label = ("> " if selected else "  ") + item.label
        color = (80, 255, 160) if selected else (150, 150, 150)
        cv2.putText(canvas, label, (16, 60 + 28 * i), font, 0.7, color, 2, cv2.LINE_AA)
    if menu.can_go_back:
        cv2.putText(canvas, "[double-swipe-left: back]", (16, 32), font, 0.5, (120, 120, 120), 1)

    if cursor is not None:
        cx, cy = int(cursor[0] * (w - 1)), int(cursor[1] * (h - 1))
        cv2.drawMarker(canvas, (cx, cy), (60, 200, 255), cv2.MARKER_CROSS, 24, 2)

    lines = [
        f"energy {state.energy:.4f}",
        f"cursor {cursor[0]:.2f},{cursor[1]:.2f}" if cursor else "cursor -",
        f"event  {last_event.name if last_event else '-'}",
        f"fps    {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (16, h - 16 - 22 * i), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA
        )
    return canvas


def run(settings: Settings) -> None:
    """Run the capture/render loop until the user quits."""
    trails = TrailsAccumulator(decay=settings.trails_decay)
    analyzer = MotionAnalyzer(
        threshold=settings.motion_threshold,
        min_energy=settings.min_energy,
        cell_size=settings.blob_cell_size,
        window_cells=settings.blob_window_cells,
        velocity_window=settings.velocity_window,
    )
    recognizer = GestureRecognizer(
        swipe_velocity_threshold=settings.swipe_velocity_threshold,
        swipe_release_factor=settings.swipe_release_factor,
        event_cooldown_s=settings.event_cooldown_s,
        double_swipe_window_s=settings.double_swipe_window_s,
        cursor_smoothing=settings.cursor_smoothing,
        select_detector=PushSelectDetector(
            energy_threshold=settings.select_energy_threshold,
            steady_speed=settings.select_steady_speed,
        ),
    )
    menu = Menu(_DEMO_MENU)
    dither_mode = settings.dither_mode
    scheme = settings.color_scheme

    prev = None
    last_event: Event | None = None
    fps = 0.0
    last_time = time.perf_counter()

    with (
        Camera(
            index=settings.camera_index,
            width=settings.frame_width,
            height=settings.frame_height,
            mirror=settings.mirror,
        ) as camera,
        JsonlLogger(settings.telemetry_path) as log,
    ):
        while True:
            frame = camera.read()
            if frame is None:
                break
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            gray = to_gray(frame)
            if prev is None:
                prev = gray
                continue
            diff = frame_difference(gray, prev, noise_floor=settings.noise_floor)
            prev = gray

            trail_signal = trails.update(diff)
            state = analyzer.analyze(diff, timestamp=now)

            for event in recognizer.update(state):
                last_event = event
                action = menu.handle_event(event)
                log.log("gesture", event=event.name, cursor=recognizer.cursor)
                if action is not None:
                    log.log("selection", action=action)
            log.log("frame", energy=state.energy, centroid=state.centroid, fps=round(fps, 2))

            grid = downsample(boost(trail_signal, settings.signal_gain), settings.cell_size)
            mask = dither(grid, dither_mode)
            canvas = render_dots(
                mask,
                cell_size=settings.cell_size,
                dot_radius=settings.dot_radius,
                foreground=scheme.foreground,
                background=scheme.background,
                brightness=grid,
            )
            display = np.asarray(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            display = _draw_hud(display, menu, state, recognizer.cursor, last_event, fps)
            cv2.imshow(_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("d"):
                dither_mode = (
                    DitherMode.FLOYD_STEINBERG
                    if dither_mode is DitherMode.BAYER
                    else DitherMode.BAYER
                )

    cv2.destroyAllWindows()


def main() -> None:
    """Console entry point: parse args and run the loop."""
    parser = argparse.ArgumentParser(prog="motioncon", description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    args = parser.parse_args()
    run(Settings(camera_index=args.camera))


if __name__ == "__main__":
    main()
