"""Application loop: camera -> difference/flow -> strokes -> menu -> display."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import numpy.typing as npt

from motioncon.config import DitherMode, Event, Settings
from motioncon.control.flick import FlickDetector
from motioncon.render.dither import dither
from motioncon.render.dots import downsample, render_dots
from motioncon.telemetry.logger import JsonlLogger
from motioncon.ui.menu import Menu, MenuItem
from motioncon.vision.camera import Camera
from motioncon.vision.difference import TrailsAccumulator, boost, frame_difference, to_gray
from motioncon.vision.flow import FlowAnalyzer, FlowState

_WINDOW = "motion-console"
_FEEDBACK_S = 2.0

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

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (90, 90, 90)


def _make_detector(settings: Settings) -> FlickDetector:
    return FlickDetector(
        presence_floor=settings.presence_floor,
        quiet_frac=settings.quiet_frac,
        settle_s=settings.settle_s,
        settle_mag=settings.settle_mag,
        arm_window_s=settings.arm_window_s,
        capture_floor=settings.capture_floor,
        burst_quiet_s=settings.burst_quiet_s,
        burst_max_s=settings.burst_max_s,
        throw_impulse=settings.throw_impulse,
        coh_min=settings.coh_min,
        refractory_s=settings.refractory_s,
    )


def _feedback_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    if raw == Event.STROKE_LEFT.name:
        return "LEFT"
    if raw == Event.STROKE_RIGHT.name:
        return "RIGHT"
    return raw


def _draw_hud(
    canvas: npt.NDArray[np.uint8],
    menu: Menu,
    state: FlowState,
    detector: FlickDetector,
    last_event: Event | None,
    last_event_time: float,
    now: float,
    fps: float,
    gesture_band: tuple[float, float],
) -> npt.NDArray[np.uint8]:
    """Draw menu, gesture band, capture balance, and classify feedback."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    menu_h = 32 + 36 * len(menu.items)

    band_top = int(gesture_band[0] * h)
    band_bottom = int(gesture_band[1] * h)
    cv2.line(canvas, (0, band_top), (w, band_top), _GRAY, 1)
    cv2.line(canvas, (0, band_bottom), (w, band_bottom), _GRAY, 1)
    cv2.putText(canvas, "GESTURE BAND", (w - 138, band_top + 16), font, 0.4, _GRAY, 1)

    cv2.rectangle(canvas, (8, 8), (280, menu_h), _WHITE, 2)
    if menu.can_go_back:
        cv2.putText(canvas, "BACK: B", (16, 30), font, 0.45, _WHITE, 1, cv2.LINE_AA)

    for i, item in enumerate(menu.items):
        y = 60 + 36 * i
        selected = i == menu.selected_index
        label = item.label.upper()
        if selected:
            cv2.rectangle(canvas, (12, y - 22), (272, y + 8), _WHITE, -1)
            cv2.putText(canvas, label, (20, y), font, 0.65, _BLACK, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, label, (20, y), font, 0.65, _WHITE, 2, cv2.LINE_AA)

    if detector.phase.name == "CAPTURE":
        phase_text = "CAPTURE"
    elif detector.ready:
        remaining = detector.arm_remaining_s
        phase_text = f"READY {remaining:.1f}s" if remaining is not None else "READY"
    else:
        phase_text = detector.phase.name
    flow_text = (
        f"FLOW {state.magnitude:.2f} | COH {state.coherence:.2f} | "
        f"ACTIVE {state.active_frac:.3f} | {phase_text}"
    )
    cv2.putText(canvas, flow_text, (12, h - 46), font, 0.45, _WHITE, 1, cv2.LINE_AA)

    balance = detector.capture_balance
    if balance is not None:
        left_frac, right_frac = balance
        bar_w, bar_h = 280, 14
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 - bar_h // 2
        mid = bar_x + bar_w // 2
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _GRAY, 1)
        left_px = int((bar_w // 2) * left_frac)
        right_px = int((bar_w // 2) * right_frac)
        if left_px > 0:
            cv2.rectangle(canvas, (mid - left_px, bar_y), (mid, bar_y + bar_h), _WHITE, -1)
        if right_px > 0:
            cv2.rectangle(canvas, (mid, bar_y), (mid + right_px, bar_y + bar_h), _WHITE, -1)
        cv2.line(canvas, (mid, bar_y - 4), (mid, bar_y + bar_h + 4), _WHITE, 1)
        cv2.putText(canvas, "L", (bar_x - 16, bar_y + 12), font, 0.45, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, "R", (bar_x + bar_w + 6, bar_y + 12), font, 0.45, _WHITE, 1, cv2.LINE_AA
        )

    feedback = _feedback_label(detector.feedback)
    if feedback is not None and now - detector.feedback_at <= _FEEDBACK_S:
        cv2.putText(
            canvas,
            feedback,
            (w // 2 - 40, h // 2 - 30),
            font,
            1.2,
            _WHITE,
            2,
            cv2.LINE_AA,
        )

    strip_h = 28
    cv2.rectangle(canvas, (0, h - strip_h), (w, h), _BLACK, -1)
    cv2.line(canvas, (0, h - strip_h), (w, h - strip_h), _WHITE, 2)
    sel = menu.selected_item.label.upper()
    visible_event = last_event if now - last_event_time <= _FEEDBACK_S else None
    status = (
        f"SEL {menu.selected_index + 1}/{len(menu.items)} {sel} | "
        f"EVT {visible_event.name if visible_event else '-'} | FPS {fps:.0f}"
    )
    cv2.putText(canvas, status, (12, h - 8), font, 0.5, _WHITE, 1, cv2.LINE_AA)
    return canvas


def run(settings: Settings) -> None:
    """Run the capture/render loop until the user quits."""
    trails = TrailsAccumulator(decay=settings.trails_decay)
    flow = FlowAnalyzer(
        width=settings.flow_width,
        height=settings.flow_height,
        mag_floor=settings.flow_mag_floor,
        gesture_band=settings.gesture_band,
        ignore_bottom=settings.ignore_bottom,
    )
    detector = _make_detector(settings)
    menu = Menu(_DEMO_MENU)
    dither_mode = settings.dither_mode
    scheme = settings.color_scheme

    prev = None
    last_event: Event | None = None
    last_event_time = float("-inf")
    fps = 0.0
    last_time = time.perf_counter()
    frame_counter = 0

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
            state = flow.analyze(gray, timestamp=now)
            if prev is None:
                prev = gray
                continue
            diff = frame_difference(gray, prev, noise_floor=settings.noise_floor)
            prev = gray

            trail_signal = trails.update(diff)

            for event in detector.update(state):
                last_event = event
                last_event_time = now
                menu.handle_event(event)
                if settings.telemetry_enabled:
                    log.log(
                        "gesture",
                        event=event.name,
                        menu_index=menu.selected_index,
                        menu_item=menu.selected_item.label,
                    )

            burst = detector.take_burst()
            if burst is not None and settings.telemetry_enabled:
                log.log(
                    "burst",
                    start_ts=burst.start_ts,
                    end_ts=burst.end_ts,
                    n_samples=burst.n_samples,
                    segments=[
                        {
                            "sign": segment.sign,
                            "integral": round(segment.integral, 6),
                            "duration": round(segment.duration, 4),
                            "mean_coherence": round(segment.mean_coherence, 4),
                        }
                        for segment in burst.segments
                    ],
                    outcome=burst.outcome,
                    y_integral=round(burst.y_integral, 6),
                )

            if settings.telemetry_enabled:
                frame_counter += 1
                if frame_counter % settings.telemetry_frame_stride == 0:
                    log.log(
                        "flow",
                        mean_flow=[round(value, 4) for value in state.mean_flow],
                        magnitude=round(state.magnitude, 4),
                        coherence=round(state.coherence, 4),
                        active_frac=round(state.active_frac, 4),
                        phase=detector.phase.name,
                        sample_timestamp=state.timestamp,
                        fps=round(fps, 2),
                    )

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
            display = _draw_hud(
                display,
                menu,
                state,
                detector,
                last_event,
                last_event_time,
                now,
                fps,
                settings.gesture_band,
            )
            cv2.imshow(_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("b"):
                menu.back()
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
