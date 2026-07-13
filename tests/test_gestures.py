"""Tests for the gesture state machine, driven by scripted MotionState streams."""

from __future__ import annotations

from motioncon.config import Event
from motioncon.control.gestures import GestureRecognizer, PushSelectDetector
from motioncon.vision.motion import MotionState


def state(
    t: float,
    velocity: tuple[float, float] | None = (0.0, 0.0),
    centroid: tuple[float, float] | None = (0.5, 0.5),
    energy: float = 0.01,
) -> MotionState:
    """Shorthand for building synthetic motion states."""
    return MotionState(timestamp=t, energy=energy, centroid=centroid, velocity=velocity)


def make_recognizer(
    swipe_velocity_threshold: float = 1.0,
    swipe_release_factor: float = 0.5,
    event_cooldown_s: float = 0.3,
    double_swipe_window_s: float = 1.0,
    cursor_smoothing: float = 0.5,
) -> GestureRecognizer:
    """A recognizer with round-number thresholds for readable tests."""
    return GestureRecognizer(
        swipe_velocity_threshold=swipe_velocity_threshold,
        swipe_release_factor=swipe_release_factor,
        event_cooldown_s=event_cooldown_s,
        double_swipe_window_s=double_swipe_window_s,
        cursor_smoothing=cursor_smoothing,
    )


class TestSwipes:
    def test_each_direction_maps_to_its_event(self) -> None:
        cases = {
            (2.0, 0.0): Event.SWIPE_RIGHT,
            (-2.0, 0.0): Event.SWIPE_LEFT,
            (0.0, 2.0): Event.SWIPE_DOWN,
            (0.0, -2.0): Event.SWIPE_UP,
        }
        for velocity, expected in cases.items():
            recognizer = make_recognizer()
            assert recognizer.update(state(0.0, velocity=velocity)) == [expected]

    def test_sustained_swipe_emits_once(self) -> None:
        recognizer = make_recognizer()
        events = []
        for i in range(5):
            events += recognizer.update(state(0.05 * i, velocity=(2.0, 0.0)))
        assert events == [Event.SWIPE_RIGHT]

    def test_rearm_after_release_and_cooldown(self) -> None:
        recognizer = make_recognizer()
        assert recognizer.update(state(0.0, velocity=(2.0, 0.0))) == [Event.SWIPE_RIGHT]
        assert recognizer.update(state(0.1, velocity=(0.1, 0.0))) == []  # released
        assert recognizer.update(state(0.5, velocity=(2.0, 0.0))) == [Event.SWIPE_RIGHT]

    def test_debounce_suppresses_rapid_repeat(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3)
        assert recognizer.update(state(0.0, velocity=(2.0, 0.0))) == [Event.SWIPE_RIGHT]
        assert recognizer.update(state(0.05, velocity=(0.0, 0.0))) == []  # released
        # Fast enough to re-trigger but still inside the cooldown window.
        assert recognizer.update(state(0.1, velocity=(2.0, 0.0))) == []

    def test_no_swipe_without_velocity(self) -> None:
        recognizer = make_recognizer()
        assert recognizer.update(state(0.0, velocity=None, centroid=None)) == []


class TestDoubleSwipeLeft:
    def test_second_left_within_window_becomes_back(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=1.0)
        assert recognizer.update(state(0.0, velocity=(-2.0, 0.0))) == [Event.SWIPE_LEFT]
        recognizer.update(state(0.2, velocity=(0.0, 0.0)))
        assert recognizer.update(state(0.5, velocity=(-2.0, 0.0))) == [Event.DOUBLE_SWIPE_LEFT]

    def test_slow_second_left_is_plain_swipe(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=1.0)
        assert recognizer.update(state(0.0, velocity=(-2.0, 0.0))) == [Event.SWIPE_LEFT]
        recognizer.update(state(0.2, velocity=(0.0, 0.0)))
        assert recognizer.update(state(2.0, velocity=(-2.0, 0.0))) == [Event.SWIPE_LEFT]

    def test_window_resets_after_double(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=10.0)
        recognizer.update(state(0.0, velocity=(-2.0, 0.0)))
        recognizer.update(state(0.2, velocity=(0.0, 0.0)))
        assert recognizer.update(state(0.5, velocity=(-2.0, 0.0))) == [Event.DOUBLE_SWIPE_LEFT]
        recognizer.update(state(0.7, velocity=(0.0, 0.0)))
        # A third left swipe starts a fresh pair rather than chaining doubles.
        assert recognizer.update(state(1.0, velocity=(-2.0, 0.0))) == [Event.SWIPE_LEFT]


class TestPushSelect:
    def test_energy_spike_fires_once(self) -> None:
        detector = PushSelectDetector(energy_threshold=0.2, steady_speed=0.5)
        assert not detector.update(state(0.0, energy=0.05))
        assert detector.update(state(0.1, energy=0.3, velocity=(0.1, 0.0)))
        assert not detector.update(state(0.2, energy=0.3, velocity=(0.1, 0.0)))  # still high
        assert not detector.update(state(0.3, energy=0.05))  # re-arms
        assert detector.update(state(0.4, energy=0.3, velocity=(0.1, 0.0)))

    def test_spike_while_moving_fast_is_not_select(self) -> None:
        detector = PushSelectDetector(energy_threshold=0.2, steady_speed=0.5)
        assert not detector.update(state(0.0, energy=0.3, velocity=(2.0, 0.0)))

    def test_recognizer_emits_select(self) -> None:
        recognizer = make_recognizer()
        recognizer_events = recognizer.update(
            state(0.0, energy=0.5, velocity=(0.0, 0.0), centroid=(0.5, 0.5))
        )
        assert recognizer_events == [Event.SELECT]


class TestSwappableDetector:
    class AlwaysFire:
        def update(self, state: MotionState) -> bool:
            return True

    def test_custom_detector_is_used_and_debounced(self) -> None:
        recognizer = GestureRecognizer(
            swipe_velocity_threshold=1.0,
            event_cooldown_s=0.3,
            select_detector=self.AlwaysFire(),
        )
        assert recognizer.update(state(0.0)) == [Event.SELECT]
        assert recognizer.update(state(0.1)) == []  # cooldown
        assert recognizer.update(state(0.5)) == [Event.SELECT]


class TestCursor:
    def test_cursor_smooths_centroid(self) -> None:
        recognizer = make_recognizer(cursor_smoothing=0.5)
        recognizer.update(state(0.0, centroid=(0.0, 0.0)))
        recognizer.update(state(0.1, centroid=(1.0, 1.0)))
        assert recognizer.cursor is not None
        assert recognizer.cursor[0] == 0.5
        assert recognizer.cursor[1] == 0.5

    def test_cursor_holds_position_when_motion_stops(self) -> None:
        recognizer = make_recognizer()
        recognizer.update(state(0.0, centroid=(0.3, 0.7)))
        recognizer.update(state(0.1, centroid=None, velocity=None))
        assert recognizer.cursor == (0.3, 0.7)

    def test_cursor_none_before_any_motion(self) -> None:
        assert make_recognizer().cursor is None
