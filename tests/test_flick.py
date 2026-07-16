"""Scripted tests for burst→quiet arming and capture→classify strokes."""

from __future__ import annotations

from motioncon.config import Event
from motioncon.control.flick import FlickDetector, FlickPhase
from motioncon.vision.flow import FlowState


def sample(
    t: float,
    flow: tuple[float, float] = (0.0, 0.0),
    *,
    coherence: float = 0.0,
    active: float = 0.10,
) -> FlowState:
    return FlowState(flow, coherence, active, t)


def quiet(t: float) -> FlowState:
    """Stillness: low activity and near-zero flow."""
    return sample(t, (0.0, 0.0), coherence=0.0, active=0.0)


def detector(**overrides: float) -> FlickDetector:
    params: dict[str, float] = {
        "presence_floor": 0.02,
        "quiet_frac": 0.02,
        "settle_s": 0.20,
        "settle_mag": 0.30,
        "arm_window_s": 3.0,
        "capture_floor": 0.15,
        "burst_quiet_s": 0.20,
        "burst_max_s": 1.0,
        "throw_impulse": 0.05,
        "coh_min": 0.50,
        "refractory_s": 0.40,
    }
    params.update(overrides)
    return FlickDetector(**params)


def arm(d: FlickDetector, start: float = 0.0) -> float:
    """Burst into the band, then remain quiet long enough to arm. Return arm time."""
    assert d.update(sample(start, (0.0, -1.5), coherence=0.9, active=0.10)) == []
    assert d.update(quiet(start + 0.05)) == []
    assert d.update(quiet(start + 0.15)) == []
    assert d.update(quiet(start + 0.26)) == []
    assert d.phase is FlickPhase.ARMED
    return start + 0.26


def run(d: FlickDetector, states: list[FlowState]) -> list[Event]:
    events: list[Event] = []
    for state in states:
        events.extend(d.update(state))
    return events


def close_burst(start: float, *, step: float = 0.05, seconds: float = 0.25) -> list[FlowState]:
    """Quiet samples long enough to end an open capture."""
    n = int(seconds / step) + 1
    return [quiet(start + i * step) for i in range(n)]


def test_clean_right_throw_fires_once() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        sample(t0 + 0.05, (2.0, 0.4), coherence=0.85, active=0.12),
        sample(t0 + 0.10, (2.0, 0.3), coherence=0.85, active=0.12),
        sample(t0 + 0.15, (2.0, 0.2), coherence=0.85, active=0.12),
        sample(t0 + 0.20, (2.0, 0.1), coherence=0.85, active=0.12),
        sample(t0 + 0.25, (2.0, 0.0), coherence=0.85, active=0.12),
        *close_burst(t0 + 0.30),
    ]
    assert run(d, states) == [Event.STROKE_RIGHT]
    burst = d.take_burst()
    assert burst is not None
    assert burst.outcome == Event.STROKE_RIGHT.name


def test_small_left_windup_then_right_throw() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        # Wind-up: left integral stays under throw_impulse.
        sample(t0 + 0.05, (-0.8, 0.0), coherence=0.8, active=0.08),
        # Main throw: right.
        sample(t0 + 0.10, (2.2, 0.5), coherence=0.9, active=0.14),
        sample(t0 + 0.15, (2.2, 0.4), coherence=0.9, active=0.14),
        sample(t0 + 0.20, (2.2, 0.2), coherence=0.9, active=0.14),
        sample(t0 + 0.25, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.30, (2.0, 0.0), coherence=0.9, active=0.14),
        *close_burst(t0 + 0.35),
    ]
    assert run(d, states) == [Event.STROKE_RIGHT]


def test_right_throw_plus_left_return_in_one_burst() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        sample(t0 + 0.05, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.10, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.15, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.20, (2.0, 0.0), coherence=0.9, active=0.14),
        # Immediate return stroke — must not fire a second event.
        sample(t0 + 0.25, (-2.5, 0.0), coherence=0.95, active=0.14),
        sample(t0 + 0.30, (-2.5, 0.0), coherence=0.95, active=0.14),
        sample(t0 + 0.35, (-2.5, 0.0), coherence=0.95, active=0.14),
        *close_burst(t0 + 0.40),
    ]
    assert run(d, states) == [Event.STROKE_RIGHT]


def test_flow_blinking_still_fires() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        sample(t0 + 0.05, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.10, (0.0, 0.0), coherence=0.0, active=0.0),  # dead
        sample(t0 + 0.15, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.20, (0.0, 0.0), coherence=0.0, active=0.0),  # dead
        sample(t0 + 0.25, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.30, (2.0, 0.0), coherence=0.9, active=0.14),
        *close_burst(t0 + 0.35),
    ]
    assert run(d, states) == [Event.STROKE_RIGHT]


def test_slow_ambiguous_drift_fires_nothing() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        sample(t0 + 0.05, (0.4, 0.1), coherence=0.7, active=0.05),
        sample(t0 + 0.10, (0.3, -0.1), coherence=0.6, active=0.04),
        sample(t0 + 0.15, (-0.35, 0.05), coherence=0.55, active=0.05),
        sample(t0 + 0.20, (0.3, 0.0), coherence=0.6, active=0.04),
        sample(t0 + 0.25, (-0.25, 0.0), coherence=0.55, active=0.04),
        *close_burst(t0 + 0.30),
    ]
    assert run(d, states) == []
    assert d.feedback == "?"
    burst = d.take_burst()
    assert burst is not None
    assert burst.outcome == "none"
    assert d.phase is FlickPhase.ARMED


def test_two_separate_bursts_classify_independently() -> None:
    d = detector()
    t0 = arm(d)
    first = [
        sample(t0 + 0.05, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.10, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.15, (2.2, 0.0), coherence=0.9, active=0.14),
        sample(t0 + 0.20, (2.0, 0.0), coherence=0.9, active=0.14),
        *close_burst(t0 + 0.25),
    ]
    assert run(d, first) == [Event.STROKE_RIGHT]
    assert d.phase is FlickPhase.FIRED
    d.take_burst()

    # Refractory, then re-arm with a fresh burst→quiet latch.
    t_ref = t0 + 0.70
    assert d.update(quiet(t_ref)) == []  # still refractory if fired ~t0+0.45
    assert d.update(quiet(t_ref + 0.20)) == []  # exit refractory → settle/entry
    # Raise then settle to re-arm.
    assert d.update(sample(t_ref + 0.25, (1.0, 0.0), coherence=0.8, active=0.12)) == []
    assert run(d, close_burst(t_ref + 0.30, seconds=0.30)) == []
    assert d.phase is FlickPhase.ARMED

    t1 = t_ref + 0.65
    second = [
        sample(t1 + 0.05, (-2.2, 0.0), coherence=0.9, active=0.14),
        sample(t1 + 0.10, (-2.2, 0.0), coherence=0.9, active=0.14),
        sample(t1 + 0.15, (-2.2, 0.0), coherence=0.9, active=0.14),
        sample(t1 + 0.20, (-2.0, 0.0), coherence=0.9, active=0.14),
        *close_burst(t1 + 0.25),
    ]
    assert run(d, second) == [Event.STROKE_LEFT]


def test_arm_window_expires_without_capture() -> None:
    d = detector(arm_window_s=0.50)
    arm(d)
    assert d.arm_remaining_s is not None
    assert run(d, [quiet(0.40), quiet(0.70), quiet(0.80)]) == []
    assert d.phase is FlickPhase.EMPTY


def test_continuous_motion_without_quiet_never_arms() -> None:
    d = detector()
    states = [
        sample(i * 0.05, (1.2, 0.0), coherence=0.9, active=0.15) for i in range(40)
    ]
    assert run(d, states) == []
    assert d.phase is FlickPhase.ENTRY


def test_clean_left_throw_fires_once() -> None:
    d = detector()
    t0 = arm(d)
    states = [
        sample(t0 + 0.05, (-2.0, 0.3), coherence=0.85, active=0.12),
        sample(t0 + 0.10, (-2.0, 0.2), coherence=0.85, active=0.12),
        sample(t0 + 0.15, (-2.0, 0.1), coherence=0.85, active=0.12),
        sample(t0 + 0.20, (-2.0, 0.0), coherence=0.85, active=0.12),
        sample(t0 + 0.25, (-2.0, 0.0), coherence=0.85, active=0.12),
        *close_burst(t0 + 0.30),
    ]
    assert run(d, states) == [Event.STROKE_LEFT]


def test_capture_balance_updates_during_throw() -> None:
    d = detector()
    t0 = arm(d)
    assert d.capture_balance is None
    assert d.update(sample(t0 + 0.05, (2.0, 0.0), coherence=0.9, active=0.12)) == []
    assert d.phase is FlickPhase.CAPTURE
    balance = d.capture_balance
    assert balance is not None
    left, right = balance
    assert right > left
