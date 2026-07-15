"""Dense-flow tests using synthetic translating image patterns."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from motioncon.vision.flow import FlowAnalyzer


def pattern(
    *,
    top: int = 28,
    left: int = 30,
    height: int = 20,
    width: int = 24,
) -> npt.NDArray[np.float32]:
    """High-contrast textured rectangle on a black 96x72 frame."""
    frame = np.zeros((72, 96), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    texture = ((xx // 3 + yy // 3) % 2).astype(np.float32)
    frame[top : top + height, left : left + width] = 0.35 + 0.65 * texture
    return frame


def shifted(frame: npt.NDArray[np.float32], dx: int, dy: int) -> npt.NDArray[np.float32]:
    """Translate without wraparound."""
    result = np.zeros_like(frame)
    src_y0, src_y1 = max(-dy, 0), min(frame.shape[0] - dy, frame.shape[0])
    src_x0, src_x1 = max(-dx, 0), min(frame.shape[1] - dx, frame.shape[1])
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    result[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return result


def analyze_translation(
    dx: int, dy: int, *, first: npt.NDArray[np.float32] | None = None
) -> tuple[float, float, float, float]:
    analyzer = FlowAnalyzer(mag_floor=0.05)
    base = pattern() if first is None else first
    analyzer.analyze(base, 0.0)
    state = analyzer.analyze(shifted(base, dx, dy), 1 / 30)
    return (*state.mean_flow, state.coherence, state.active_frac)


def test_right_translation_has_coherent_positive_x_flow() -> None:
    x, y, coherence, active_frac = analyze_translation(3, 0)
    assert x > 1.0
    assert abs(y) < abs(x) * 0.25
    assert coherence > 0.7
    assert active_frac > 0.02


def test_up_translation_has_coherent_negative_y_flow() -> None:
    x, y, coherence, _ = analyze_translation(0, -3)
    assert y < -1.0
    assert abs(x) < abs(y) * 0.25
    assert coherence > 0.7


def test_motion_above_gesture_band_is_excluded() -> None:
    head_pattern = pattern(top=2, height=12)
    x, y, coherence, active_frac = analyze_translation(4, 0, first=head_pattern)
    assert (x, y, coherence, active_frac) == (0.0, 0.0, 0.0, 0.0)


def test_motion_in_ignored_bottom_is_excluded() -> None:
    keyboard_pattern = pattern(top=64, height=7)
    x, y, coherence, active_frac = analyze_translation(3, 0, first=keyboard_pattern)
    assert (x, y, coherence, active_frac) == (0.0, 0.0, 0.0, 0.0)


def test_first_frame_is_empty_and_reset_forgets_history() -> None:
    analyzer = FlowAnalyzer()
    assert analyzer.analyze(pattern(), 0.0).active_frac == 0.0
    analyzer.reset()
    assert analyzer.analyze(pattern(left=34), 0.1).active_frac == 0.0
