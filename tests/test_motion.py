"""Tests for pure motion analysis with synthetic difference signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from motioncon.vision.motion import MotionAnalyzer, MotionState, score_candidate


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
        analyzer = MotionAnalyzer(cell_size=16, min_track_area=0.0)
        state = analyzer.analyze(blob_frame((100, 100), center=(30, 70), radius=6), timestamp=0.0)

        assert state.has_motion
        assert state.centroid is not None
        cx, cy = state.centroid
        np.testing.assert_allclose(cx, 70 / 99, atol=0.02)
        np.testing.assert_allclose(cy, 30 / 99, atol=0.02)

    def test_dominant_blob_wins_over_noise(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, min_track_area=0.0)
        signal = blob_frame((128, 128), center=(90, 90), radius=10, value=1.0)
        signal += blob_frame((128, 128), center=(10, 10), radius=2, value=0.5)
        state = analyzer.analyze(np.clip(signal, 0.0, 1.0), timestamp=0.0)

        assert state.centroid is not None
        cx, cy = state.centroid
        np.testing.assert_allclose(cx, 90 / 127, atol=0.05)
        np.testing.assert_allclose(cy, 90 / 127, atol=0.05)

    def test_empty_frame_has_no_motion(self) -> None:
        analyzer = MotionAnalyzer(min_track_area=0.0)
        state = analyzer.analyze(np.zeros((64, 64), dtype=np.float32), timestamp=0.0)
        assert not state.has_motion
        assert state.centroid is None
        assert state.velocity is None
        assert state.energy == 0.0

    def test_subthreshold_signal_ignored(self) -> None:
        analyzer = MotionAnalyzer(threshold=0.5)
        state = analyzer.analyze(np.full((64, 64), 0.2, dtype=np.float32), timestamp=0.0)
        assert not state.has_motion


class TestSpatialScoring:
    def test_left_blob_beats_center(self) -> None:
        left = score_candidate(0.2, 0.4, 0.2, 0.0)
        center = score_candidate(0.5, 0.4, 0.2, 0.0)
        assert left > center

    def test_bottom_blob_penalized(self) -> None:
        side = score_candidate(0.2, 0.4, 0.2, 0.0)
        bottom = score_candidate(0.5, 0.9, 0.2, 0.0)
        assert side > bottom

    def test_center_intent_override(self) -> None:
        passive = score_candidate(0.5, 0.4, 0.15, 0.0, min_track_area=0.10)
        intentional = score_candidate(0.5, 0.4, 0.15, 2.0, min_track_area=0.10)
        assert intentional > passive

    def test_left_hand_preferred_over_center_head(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, min_track_area=0.08)
        signal = blob_frame((128, 128), center=(64, 64), radius=14, value=1.0)
        signal += blob_frame((128, 128), center=(64, 20), radius=14, value=1.0)
        state = analyzer.analyze(np.clip(signal, 0.0, 1.0), timestamp=0.0)
        assert state.centroid is not None
        assert state.centroid[0] < 0.45

    def test_bottom_keyboard_loses_to_side_hand(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, min_track_area=0.08)
        signal = blob_frame((128, 128), center=(120, 64), radius=14, value=1.0)
        signal += blob_frame((128, 128), center=(64, 20), radius=14, value=1.0)
        state = analyzer.analyze(np.clip(signal, 0.0, 1.0), timestamp=0.0)
        assert state.centroid is not None
        assert state.centroid[1] < 0.6


class TestHardLock:
    def test_hard_lock_holds_through_distraction(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16,
            min_track_area=0.08,
            lock_duration_s=2.0,
            track_search_radius=0.35,
        )
        hand = blob_frame((128, 128), center=(64, 20), radius=14, value=1.0)
        locked = analyzer.analyze(hand, timestamp=0.0)
        assert locked.has_motion and locked.centroid is not None
        assert locked.track_locked
        hand_x = locked.centroid[0]

        distracted = hand.copy()
        distracted += blob_frame((128, 128), center=(120, 110), radius=14, value=1.0)
        state = analyzer.analyze(np.clip(distracted, 0.0, 1.0), timestamp=0.5)
        assert state.has_motion and state.centroid is not None
        assert state.track_locked
        assert abs(state.centroid[0] - hand_x) < 0.12

    def test_jump_cap_limits_teleport(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16,
            min_track_area=0.0,
            max_centroid_step=0.08,
            lock_duration_s=2.0,
        )
        analyzer.analyze(blob_frame((128, 128), center=(64, 64), radius=10), timestamp=0.0)
        prior = analyzer.analyze(
            blob_frame((128, 128), center=(72, 64), radius=10), timestamp=0.05
        )
        state = analyzer.analyze(
            blob_frame((128, 128), center=(110, 64), radius=10), timestamp=0.1
        )
        assert prior.centroid is not None and state.centroid is not None
        step = abs(state.centroid[0] - prior.centroid[0])
        assert step <= 0.081

    def test_reacquires_after_lock_expires(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16,
            min_track_area=0.08,
            lock_duration_s=0.5,
            track_max_miss_frames=2,
        )
        analyzer.analyze(blob_frame((128, 128), center=(64, 64), radius=14), timestamp=0.0)
        analyzer.analyze(np.zeros((128, 128), dtype=np.float32), timestamp=0.6)
        state = analyzer.analyze(
            blob_frame((128, 128), center=(20, 20), radius=14), timestamp=0.7
        )
        assert state.has_motion and state.centroid is not None
        assert state.centroid[0] < 0.35

    def test_lock_refreshes_while_hand_tracked(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16,
            min_track_area=0.08,
            lock_duration_s=0.5,
        )
        hand = blob_frame((128, 128), center=(64, 20), radius=14, value=1.0)
        analyzer.analyze(hand, timestamp=0.0)
        state = analyzer.analyze(hand, timestamp=1.0)
        assert state.has_motion
        assert state.track_locked
        assert state.lock_remaining_s > 0.0

    def test_coasts_through_brief_dropout(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16,
            min_track_area=0.08,
            track_max_miss_frames=3,
        )
        hand = blob_frame((128, 128), center=(64, 20), radius=14, value=1.0)
        locked = analyzer.analyze(hand, timestamp=0.0)
        assert locked.centroid is not None
        empty = analyzer.analyze(np.zeros((128, 128), dtype=np.float32), timestamp=0.05)
        assert empty.has_motion
        assert empty.centroid == locked.centroid


class TestEnergy:
    def test_energy_is_mean_of_active_signal(self) -> None:
        analyzer = MotionAnalyzer(threshold=0.1)
        signal = np.zeros((10, 10), dtype=np.float32)
        signal[:5, :] = 1.0
        state = analyzer.analyze(signal, timestamp=0.0)
        assert state.energy == 0.5


class TestBlobArea:
    def test_larger_blob_has_larger_area(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, window_cells=5, min_track_area=0.0)
        small = analyzer.analyze(
            blob_frame((128, 128), center=(64, 64), radius=4), timestamp=0.0
        )
        analyzer.reset()
        large = analyzer.analyze(
            blob_frame((128, 128), center=(64, 64), radius=16), timestamp=0.0
        )
        assert small.has_motion and large.has_motion
        assert large.blob_area > small.blob_area

    def test_empty_frame_has_zero_area(self) -> None:
        analyzer = MotionAnalyzer(min_track_area=0.0)
        state = analyzer.analyze(np.zeros((64, 64), dtype=np.float32), timestamp=0.0)
        assert state.blob_area == 0.0

    def test_sub_palm_blob_ignored(self) -> None:
        analyzer = MotionAnalyzer(min_track_area=0.10, cell_size=16)
        state = analyzer.analyze(
            blob_frame((128, 128), center=(64, 64), radius=5), timestamp=0.0
        )
        assert not state.has_motion
        assert state.blob_area > 0.0
        assert state.blob_area < 0.10

    def test_palm_sized_blob_is_tracked(self) -> None:
        analyzer = MotionAnalyzer(min_track_area=0.10, cell_size=16)
        state = analyzer.analyze(
            blob_frame((128, 128), center=(64, 64), radius=14), timestamp=0.0
        )
        assert state.has_motion
        assert state.blob_area >= 0.10

    def test_raw_area_reported_when_not_tracked(self) -> None:
        analyzer = MotionAnalyzer(min_track_area=0.50, cell_size=16)
        state = analyzer.analyze(
            blob_frame((128, 128), center=(64, 64), radius=10), timestamp=0.0
        )
        assert not state.has_motion
        assert state.blob_area > 0.0


class TestVelocity:
    def test_rightward_drift(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16, velocity_window=5, min_track_area=0.0, max_centroid_step=1.0
        )
        states: list[MotionState] = []
        for i in range(5):
            frame = blob_frame((100, 100), center=(50, 20 + 10 * i), radius=5)
            states.append(analyzer.analyze(frame, timestamp=0.1 * i))

        assert states[0].velocity is None
        velocity = states[-1].velocity
        assert velocity is not None
        vx, vy = velocity
        np.testing.assert_allclose(vx, (40 / 99) / 0.4, atol=0.1)
        np.testing.assert_allclose(vy, 0.0, atol=0.05)
        assert vx > 0

    def test_upward_drift_is_negative_y(self) -> None:
        analyzer = MotionAnalyzer(cell_size=16, min_track_area=0.0)
        for i in range(4):
            frame = blob_frame((100, 100), center=(80 - 15 * i, 50), radius=5)
            state = analyzer.analyze(frame, timestamp=0.1 * i)
        assert state.velocity is not None
        assert state.velocity[1] < 0

    def test_history_resets_when_motion_lost(self) -> None:
        analyzer = MotionAnalyzer(
            cell_size=16, min_track_area=0.0, lock_duration_s=0.0, track_max_miss_frames=0
        )
        analyzer.analyze(blob_frame((100, 100), center=(50, 20), radius=5), timestamp=0.0)
        analyzer.analyze(np.zeros((100, 100), dtype=np.float32), timestamp=0.1)
        state = analyzer.analyze(blob_frame((100, 100), center=(50, 80), radius=5), timestamp=0.2)
        assert state.velocity is None
