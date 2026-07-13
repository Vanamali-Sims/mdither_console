"""Tests for pure motion analysis with synthetic difference signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from motioncon.vision.motion import MotionAnalyzer, MotionState


def blob_frame(
    shape: tuple[int, int], center: tuple[int, int], radius: int, value: float = 1.0
) -> npt.NDArray[np.float32]:
    """A synthetic difference signal: one bright circular blob."""
    frame = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    frame[(yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2] = value
    return frame


class TestCentroid:
    def test_centroid_matches_blob_position(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16)
        state = analyzer.analyze(blob_frame((100, 100), center=(30, 70), radius=6), timestamp=0.0)

        assert state.has_motion
        assert state.centroid is not None
        cx, cy = state.centroid
        np.testing.assert_allclose(cx, 70 / 99, atol=0.02)
        np.testing.assert_allclose(cy, 30 / 99, atol=0.02)

    def test_dominant_blob_wins_over_noise(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16)
        signal = blob_frame((128, 128), center=(90, 90), radius=10, value=1.0)
        # A small, weak noise blob far away must not drag the centroid.
        signal += blob_frame((128, 128), center=(10, 10), radius=2, value=0.5)
        state = analyzer.analyze(np.clip(signal, 0.0, 1.0), timestamp=0.0)

        assert state.centroid is not None
        cx, cy = state.centroid
        np.testing.assert_allclose(cx, 90 / 127, atol=0.05)
        np.testing.assert_allclose(cy, 90 / 127, atol=0.05)

    def test_empty_frame_has_no_motion(self) -> None:
        analyzer = MotionAnalyzer()
        state = analyzer.analyze(np.zeros((64, 64), dtype=np.float32), timestamp=0.0)
        assert not state.has_motion
        assert state.centroid is None
        assert state.velocity is None
        assert state.energy == 0.0

    def test_subthreshold_signal_ignored(self) -> None:
        analyzer = MotionAnalyzer(threshold=0.5)
        state = analyzer.analyze(np.full((64, 64), 0.2, dtype=np.float32), timestamp=0.0)
        assert not state.has_motion


class TestEnergy:
    def test_energy_is_mean_of_active_signal(self) -> None:
        analyzer = MotionAnalyzer(threshold=0.1)
        signal = np.zeros((10, 10), dtype=np.float32)
        signal[:5, :] = 1.0  # half the frame moving at full strength
        state = analyzer.analyze(signal, timestamp=0.0)
        assert state.energy == 0.5


class TestVelocity:
    def test_rightward_drift(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, velocity_window=5)
        states: list[MotionState] = []
        for i in range(5):
            frame = blob_frame((100, 100), center=(50, 20 + 10 * i), radius=5)
            states.append(analyzer.analyze(frame, timestamp=0.1 * i))

        assert states[0].velocity is None  # needs at least two samples
        velocity = states[-1].velocity
        assert velocity is not None
        vx, vy = velocity
        # 40 px over 0.4 s in a 100-wide frame: ~1.01 normalized units/s.
        np.testing.assert_allclose(vx, (40 / 99) / 0.4, atol=0.1)
        np.testing.assert_allclose(vy, 0.0, atol=0.05)
        assert vx > 0

    def test_upward_drift_is_negative_y(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16)
        for i in range(4):
            frame = blob_frame((100, 100), center=(80 - 15 * i, 50), radius=5)
            state = analyzer.analyze(frame, timestamp=0.1 * i)
        assert state.velocity is not None
        assert state.velocity[1] < 0

    def test_history_resets_when_motion_lost(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16)
        analyzer.analyze(blob_frame((100, 100), center=(50, 20), radius=5), timestamp=0.0)
        analyzer.analyze(np.zeros((100, 100), dtype=np.float32), timestamp=0.1)
        state = analyzer.analyze(blob_frame((100, 100), center=(50, 80), radius=5), timestamp=0.2)
        # After the gap the history restarted, so no velocity yet.
        assert state.velocity is None
