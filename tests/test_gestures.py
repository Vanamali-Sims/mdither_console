"""Tests for the gesture state machine, driven by scripted MotionState streams."""

from __future__ import annotations

from motioncon.config import Event
from motioncon.control.gestures import GestureRecognizer, SizeGrowSelectDetector
from motioncon.vision.motion import MotionState


def state(
    t: float,
    velocity: tuple[float, float] | None = (0.0, 0.0),
    centroid: tuple[float, float] | None = (0.5, 0.5),
    energy: float = 0.01,
    blob_area: float = 0.15,
    track_locked: bool = True,
) -> MotionState:
    """Shorthand for building synthetic motion states."""
    return MotionState(
        timestamp=t,
        energy=energy,
        centroid=centroid,
        velocity=velocity,
        blob_area=blob_area,
        track_locked=track_locked,
        lock_remaining_s=1.0 if track_locked else 0.0,
    )


def make_recognizer(
    swipe_intent_speed: float = 0.08,
    swipe_release_factor: float = 0.5,
    swipe_axis_dominance: float = 1.5,
    swipe_max_duration_s: float = 3.0,
    event_cooldown_s: float = 0.3,
    select_cooldown_s: float = 0.3,
    opposite_lockout_s: float = 1.0,
    double_swipe_window_s: float = 1.0,
    cursor_smoothing: float = 1.0,
    min_track_area: float = 0.0,
    swipe_min_travel: float = 0.33,
    select_steady_speed: float = 0.6,
    select_detector: SizeGrowSelectDetector | None = None,
) -> GestureRecognizer:
    """A recognizer with round-number thresholds for readable tests."""
    return GestureRecognizer(
        swipe_intent_speed=swipe_intent_speed,
        swipe_release_factor=swipe_release_factor,
        swipe_axis_dominance=swipe_axis_dominance,
        swipe_max_duration_s=swipe_max_duration_s,
        event_cooldown_s=event_cooldown_s,
        select_cooldown_s=select_cooldown_s,
        opposite_lockout_s=opposite_lockout_s,
        double_swipe_window_s=double_swipe_window_s,
        cursor_smoothing=cursor_smoothing,
        min_track_area=min_track_area,
        swipe_min_travel=swipe_min_travel,
        select_steady_speed=select_steady_speed,
        select_detector=select_detector,
    )


def stroke_states(
    velocity: tuple[float, float],
    *,
    axis: str,
    sign: int,
    start: float = 0.1,
    step: float = 0.08,
    steps: int = 8,
    fixed: float = 0.5,
    time_offset: float = 0.0,
    blob_area: float = 0.15,
    dt: float = 0.05,
) -> list[MotionState]:
    """Build a swipe stroke that travels far enough to register."""
    states: list[MotionState] = []
    for i in range(steps):
        pos = start + step * i
        centroid = (
            (pos, fixed) if axis == "x" else (fixed, pos)
        ) if sign > 0 else (
            (1.0 - pos, fixed) if axis == "x" else (fixed, 1.0 - pos)
        )
        states.append(
            state(
                time_offset + dt * i,
                velocity=velocity,
                centroid=centroid,
                blob_area=blob_area,
            )
        )
    return states


def _release(
    recognizer: GestureRecognizer,
    t: float,
    centroid: tuple[float, float] | None = None,
) -> None:
    """Simulate hand pause so the recognizer re-arms after a swipe."""
    hold = centroid if centroid is not None else recognizer.cursor
    if hold is None:
        hold = (0.5, 0.5)
    recognizer.update(state(t, velocity=(0.0, 0.0), centroid=hold))


def run_stroke(recognizer: GestureRecognizer, states: list[MotionState]) -> list[Event]:
    events: list[Event] = []
    for s in states:
        events.extend(recognizer.update(s))
    return events


class TestSwipes:
    def test_each_direction_maps_to_its_event(self) -> None:
        cases = {
            (2.0, 0.0, "x", 1): Event.SWIPE_RIGHT,
            (-2.0, 0.0, "x", -1): Event.SWIPE_LEFT,
            (0.0, 2.0, "y", 1): Event.SWIPE_DOWN,
            (0.0, -2.0, "y", -1): Event.SWIPE_UP,
        }
        for (vx, vy, axis, sign), expected in cases.items():
            recognizer = make_recognizer()
            events = run_stroke(
                recognizer, stroke_states((vx, vy), axis=axis, sign=sign)
            )
            assert expected in events

    def test_slow_realistic_stroke_fires(self) -> None:
        """Mirrors telemetry: ~0.2 norm/s, 0.08/frame steps, 33% travel."""
        recognizer = make_recognizer(swipe_intent_speed=0.08, cursor_smoothing=1.0)
        events = run_stroke(
            recognizer,
            stroke_states((0.2, 0.0), axis="x", sign=1, step=0.08, steps=8, dt=0.14),
        )
        assert events.count(Event.SWIPE_RIGHT) == 1

    def test_incomplete_travel_times_out(self) -> None:
        recognizer = make_recognizer(
            swipe_max_duration_s=0.2,
            swipe_intent_speed=0.05,
            cursor_smoothing=1.0,
        )
        events = run_stroke(
            recognizer,
            stroke_states(
                (0.2, 0.0),
                axis="x",
                sign=1,
                start=0.1,
                step=0.02,
                steps=8,
                dt=0.1,
            ),
        )
        assert Event.SWIPE_RIGHT not in events

    def test_short_stroke_does_not_register(self) -> None:
        recognizer = make_recognizer()
        events = run_stroke(
            recognizer,
            stroke_states((2.0, 0.0), axis="x", sign=1, start=0.4, step=0.02, steps=3),
        )
        assert events == []

    def test_sustained_swipe_emits_once(self) -> None:
        recognizer = make_recognizer()
        events = run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1, steps=8)
        )
        assert events.count(Event.SWIPE_RIGHT) == 1

    def test_rearm_after_release_and_cooldown(self) -> None:
        recognizer = make_recognizer(opposite_lockout_s=0.0)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            [
                state(0.55, velocity=(2.0, 0.0), centroid=(0.1, 0.5)),
                *stroke_states(
                    (2.0, 0.0), axis="x", sign=1, start=0.1, time_offset=0.6
                ),
            ],
        ) == [Event.SWIPE_RIGHT]

    def test_debounce_suppresses_rapid_repeat(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states((2.0, 0.0), axis="x", sign=1, start=0.2, steps=5),
        ) == []

    def test_no_swipe_without_trackable_hand(self) -> None:
        recognizer = make_recognizer()
        assert recognizer.update(state(0.0, velocity=None, centroid=None)) == []

    def test_sub_palm_motion_ignored(self) -> None:
        recognizer = make_recognizer(min_track_area=0.02)
        assert (
            run_stroke(
                recognizer,
                stroke_states((2.0, 0.0), axis="x", sign=1, blob_area=0.01),
            )
            == []
        )
        assert recognizer.cursor is None

    def test_gestures_require_sufficient_area(self) -> None:
        recognizer = make_recognizer(min_track_area=0.10)
        assert (
            run_stroke(
                recognizer,
                stroke_states((2.0, 0.0), axis="x", sign=1, blob_area=0.05),
            )
            == []
        )
        assert recognizer.cursor is None

    def test_diagonal_rejected_without_axis_dominance(self) -> None:
        recognizer = make_recognizer(swipe_axis_dominance=1.5)
        assert run_stroke(
            recognizer, stroke_states((2.0, 1.6), axis="x", sign=1)
        ) == []

    def test_axis_dominance_allows_clear_direction(self) -> None:
        recognizer = make_recognizer(swipe_axis_dominance=1.5)
        assert Event.SWIPE_RIGHT in run_stroke(
            recognizer, stroke_states((2.0, 1.0), axis="x", sign=1)
        )

    def test_select_blocked_during_swipe_stroke(self) -> None:
        class AlwaysFire:
            def update(self, state: MotionState) -> bool:
                return True

        recognizer = make_recognizer(
            select_cooldown_s=0.0,
            select_detector=AlwaysFire(),  # type: ignore[arg-type]
        )
        events = run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        )
        assert Event.SWIPE_RIGHT in events
        assert Event.SELECT not in events


class TestOppositeLockout:
    def test_opposite_direction_suppressed_inside_lockout(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3, opposite_lockout_s=1.0)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer, stroke_states((-2.0, 0.0), axis="x", sign=-1, start=0.6)
        ) == []

    def test_opposite_allowed_after_lockout(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3, opposite_lockout_s=1.0)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states(
                (-2.0, 0.0), axis="x", sign=-1, start=0.1, time_offset=1.3
            ),
        ) == [Event.SWIPE_LEFT]

    def test_same_direction_allowed_after_cooldown(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3, opposite_lockout_s=1.0)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states(
                (2.0, 0.0), axis="x", sign=1, start=0.1, time_offset=0.6
            ),
        ) == [Event.SWIPE_RIGHT]

    def test_orthogonal_allowed_after_cooldown(self) -> None:
        recognizer = make_recognizer(event_cooldown_s=0.3, opposite_lockout_s=1.0)
        assert run_stroke(
            recognizer, stroke_states((2.0, 0.0), axis="x", sign=1)
        ) == [Event.SWIPE_RIGHT]
        _release(recognizer, 0.45)
        assert Event.SWIPE_DOWN in run_stroke(
            recognizer,
            stroke_states(
                (0.0, 2.0), axis="y", sign=1, start=0.1, time_offset=0.6
            ),
        )


class TestDoubleSwipeLeft:
    def test_second_left_within_window_becomes_back(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=1.0, opposite_lockout_s=0.0)
        assert run_stroke(
            recognizer, stroke_states((-2.0, 0.0), axis="x", sign=-1, start=0.6)
        ) == [Event.SWIPE_LEFT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states(
                (-2.0, 0.0), axis="x", sign=-1, start=0.6, time_offset=0.75
            ),
        ) == [Event.DOUBLE_SWIPE_LEFT]

    def test_slow_second_left_is_plain_swipe(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=1.0, opposite_lockout_s=0.0)
        assert run_stroke(
            recognizer, stroke_states((-2.0, 0.0), axis="x", sign=-1, start=0.6)
        ) == [Event.SWIPE_LEFT]
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states(
                (-2.0, 0.0), axis="x", sign=-1, start=0.1, time_offset=2.0
            ),
        ) == [Event.SWIPE_LEFT]

    def test_window_resets_after_double(self) -> None:
        recognizer = make_recognizer(double_swipe_window_s=10.0, opposite_lockout_s=0.0)
        run_stroke(recognizer, stroke_states((-2.0, 0.0), axis="x", sign=-1, start=0.6))
        _release(recognizer, 0.45)
        assert run_stroke(
            recognizer,
            stroke_states(
                (-2.0, 0.0), axis="x", sign=-1, start=0.6, time_offset=0.75
            ),
        ) == [Event.DOUBLE_SWIPE_LEFT]
        _release(recognizer, 1.15)
        assert run_stroke(
            recognizer,
            stroke_states(
                (-2.0, 0.0), axis="x", sign=-1, start=0.6, time_offset=1.45
            ),
        ) == [Event.SWIPE_LEFT]


class TestSizeGrowSelect:
    def test_area_growth_fires_once(self) -> None:
        detector = SizeGrowSelectDetector(
            area_growth=1.45, min_area=0.002, steady_speed=0.5, history=4
        )
        assert not detector.update(state(0.0, blob_area=0.01, velocity=(0.0, 0.0)))
        assert not detector.update(state(0.05, blob_area=0.01, velocity=(0.0, 0.0)))
        assert not detector.update(state(0.1, blob_area=0.012, velocity=(0.0, 0.0)))
        assert detector.update(state(0.15, blob_area=0.02, velocity=(0.0, 0.0)))
        assert not detector.update(state(0.2, blob_area=0.021, velocity=(0.0, 0.0)))

    def test_growth_while_moving_fast_is_not_select(self) -> None:
        detector = SizeGrowSelectDetector(
            area_growth=1.45, min_area=0.002, steady_speed=0.5, history=3
        )
        detector.update(state(0.0, blob_area=0.01, velocity=(0.0, 0.0)))
        detector.update(state(0.05, blob_area=0.01, velocity=(0.0, 0.0)))
        assert not detector.update(state(0.1, blob_area=0.02, velocity=(2.0, 0.0)))

    def test_rearms_after_area_shrinks(self) -> None:
        detector = SizeGrowSelectDetector(
            area_growth=1.45, min_area=0.002, steady_speed=0.5, history=3
        )
        detector.update(state(0.0, blob_area=0.01))
        detector.update(state(0.05, blob_area=0.01))
        assert detector.update(state(0.1, blob_area=0.02))
        # Pull back toward the baseline to re-arm.
        detector.update(state(0.15, blob_area=0.01))
        detector.update(state(0.2, blob_area=0.01))
        detector.update(state(0.25, blob_area=0.01))
        assert detector.update(state(0.3, blob_area=0.02))

    def test_recognizer_emits_select(self) -> None:
        detector = SizeGrowSelectDetector(
            area_growth=1.45, min_area=0.002, steady_speed=0.5, history=3
        )
        recognizer = make_recognizer(select_detector=detector)
        recognizer.update(state(0.0, blob_area=0.01, velocity=(0.0, 0.0)))
        recognizer.update(state(0.05, blob_area=0.01, velocity=(0.0, 0.0)))
        assert recognizer.update(state(0.1, blob_area=0.02, velocity=(0.0, 0.0))) == [
            Event.SELECT
        ]


class TestSwappableDetector:
    class AlwaysFire:
        def update(self, state: MotionState) -> bool:
            return True

    def test_custom_detector_is_used_and_debounced(self) -> None:
        recognizer = GestureRecognizer(
            swipe_intent_speed=0.08,
            event_cooldown_s=0.3,
            select_cooldown_s=0.3,
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

    def test_cursor_clears_when_motion_stops(self) -> None:
        recognizer = make_recognizer()
        recognizer.update(state(0.0, centroid=(0.3, 0.7), blob_area=0.03))
        recognizer.update(state(0.1, centroid=None, velocity=None, blob_area=0.0))
        assert recognizer.cursor is None

    def test_cursor_none_before_any_motion(self) -> None:
        assert make_recognizer().cursor is None

    def test_swipe_travel_exposed_during_stroke(self) -> None:
        recognizer = make_recognizer(cursor_smoothing=1.0, swipe_intent_speed=0.05)
        run_stroke(
            recognizer,
            stroke_states((0.2, 0.0), axis="x", sign=1, step=0.08, steps=3),
        )
        assert recognizer.swipe_travel > 0.0
