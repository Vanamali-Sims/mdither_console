"""Scripted tests for the burst→quiet directional flick detector."""

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
    """Stillness after a burst: low activity and near-zero flow."""
    return sample(t, (0.0, 0.0), coherence=0.0, active=0.0)


def detector(**overrides: float) -> FlickDetector:
    params: dict[str, float] = {
        "presence_floor": 0.02,
        "quiet_frac": 0.02,
        "settle_s": 0.20,
        "settle_mag": 0.30,
        "arm_window_s": 3.0,
        "coh_min": 0.60,
        "coherence_collapse": 0.35,
        "flick_mag": 0.75,
        "axis_dominance": 1.8,
        "impulse_thresh": 3.0,
        "stroke_max_s": 0.50,
        "refractory_s": 0.40,
        "opp_lockout_s": 0.70,
    }
    params.update(overrides)
    return FlickDetector(**params)


def arm(d: FlickDetector, start: float = 0.0) -> None:
    """Burst into the band, then remain quiet long enough to arm."""
    assert d.update(sample(start, (0.0, -1.5), coherence=0.9, active=0.10)) == []
    assert d.update(quiet(start + 0.05)) == []
    assert d.update(quiet(start + 0.15)) == []
    assert d.update(quiet(start + 0.26)) == []
    assert d.phase is FlickPhase.ARMED


def run(d: FlickDetector, states: list[FlowState]) -> list[Event]:
    events: list[Event] = []
    for state in states:
        events.extend(d.update(state))
    return events


def test_burst_quiet_upward_stroke_fires_exactly_once() -> None:
    d = detector(settle_s=0.25)
    assert d.update(sample(0.0, (0.0, -1.5), coherence=0.9, active=0.12)) == []
    assert d.phase is FlickPhase.ENTRY
    assert d.update(quiet(0.05)) == []
    assert d.phase is FlickPhase.SETTLE
    assert d.update(quiet(0.20)) == []
    assert d.update(quiet(0.31)) == []
    assert d.phase is FlickPhase.ARMED
    events = run(
        d,
        [
            sample(0.36, (0.0, -1.7), coherence=0.9),
            sample(0.41, (0.0, -1.7), coherence=0.9),
            sample(0.46, (0.0, -1.7), coherence=0.9),
        ],
    )
    assert events == [Event.FLICK_UP]


def test_arm_window_expires_without_stroke() -> None:
    d = detector(arm_window_s=0.50)
    arm(d)
    assert d.arm_remaining_s is not None
    assert run(d, [quiet(0.40), quiet(0.70), quiet(0.80)]) == []
    assert d.phase is FlickPhase.EMPTY


def test_continuous_motion_without_quiet_never_arms() -> None:
    d = detector()
    states = [
        sample(i * 0.05, (0.0, -1.2), coherence=0.9, active=0.15) for i in range(40)
    ]
    assert run(d, states) == []
    assert d.phase is FlickPhase.ENTRY
    assert d.phase is not FlickPhase.ARMED


def test_clean_flick_up_fires_exactly_once() -> None:
    d = detector()
    arm(d)
    events = run(
        d,
        [
            sample(0.31, (0.0, -1.7), coherence=0.9),
            sample(0.36, (0.0, -1.7), coherence=0.9),
            sample(0.41, (0.0, -1.7), coherence=0.9),
        ],
    )
    assert events == [Event.FLICK_UP]


def test_flick_and_return_stroke_fires_once() -> None:
    d = detector()
    arm(d)
    states = [
        sample(0.31, (0.0, -1.7), coherence=0.9),
        sample(0.36, (0.0, -1.7), coherence=0.9),
        sample(0.41, (0.0, 2.0), coherence=0.95),
        sample(0.46, (0.0, 2.0), coherence=0.95),
        quiet(0.77),
        quiet(0.98),
        sample(1.00, (0.0, 2.0), coherence=0.95),
        sample(1.05, (0.0, 2.0), coherence=0.95),
    ]
    assert run(d, states) == [Event.FLICK_UP]


def test_continuous_low_coherence_typing_jitter_fires_nothing() -> None:
    d = detector()
    states = [sample(i * 0.05, (0.9 if i % 2 else -0.9, 0.7), coherence=0.2) for i in range(30)]
    assert run(d, states) == []
    assert d.phase is not FlickPhase.ARMED


def test_horizontal_left_stroke_fires_swipe_left() -> None:
    d = detector()
    arm(d)
    events = run(
        d,
        [
            sample(0.31, (-1.6, 0.1), coherence=0.9),
            sample(0.36, (-1.6, 0.1), coherence=0.9),
        ],
    )
    assert events == [Event.SWIPE_LEFT]


def test_refractory_blocks_an_immediate_second_flick() -> None:
    d = detector()
    arm(d)
    assert run(
        d,
        [
            sample(0.31, (0.0, -1.6), coherence=0.9),
            sample(0.36, (0.0, -1.6), coherence=0.9),
        ],
    ) == [Event.FLICK_UP]
    assert (
        run(
            d,
            [
                sample(0.45, (0.0, -2.0), coherence=0.9),
                sample(0.55, (0.0, -2.0), coherence=0.9),
                sample(0.65, (0.0, -2.0), coherence=0.9),
            ],
        )
        == []
    )
    assert d.phase is FlickPhase.FIRED


def test_raise_into_band_alone_fires_nothing() -> None:
    d = detector()
    states = [
        sample(0.0, (0.0, -1.5), coherence=0.9),
        sample(0.05, (0.0, -1.5), coherence=0.9),
        sample(0.10, (0.0, -1.5), coherence=0.9),
        quiet(0.15),
    ]
    assert run(d, states) == []
    assert d.phase is FlickPhase.SETTLE


def test_raise_settle_flick_up_fires_once() -> None:
    d = detector()
    arm(d)
    assert run(
        d,
        [
            sample(0.31, (0.0, -1.6), coherence=0.9),
            sample(0.36, (0.0, -1.6), coherence=0.9),
        ],
    ) == [Event.FLICK_UP]


def test_hand_lowered_mid_stroke_aborts_cleanly() -> None:
    d = detector()
    arm(d)
    assert d.update(sample(0.31, (0.0, -1.5), coherence=0.9)) == []
    assert d.impulse == -1.5
    # Zero flow collapses coherence; abort back to ENTRY (not presence-gated).
    assert d.update(sample(0.36, active=0.0)) == []
    assert d.phase is FlickPhase.ENTRY
    assert d.impulse == 0.0


def test_diagonal_stroke_does_not_lock_an_axis() -> None:
    d = detector()
    arm(d)
    assert d.update(sample(0.31, (-1.0, -0.9), coherence=0.95)) == []
    assert d.phase is FlickPhase.ARMED


def test_rightward_stroke_has_no_v1_event() -> None:
    d = detector()
    arm(d)
    events = run(
        d,
        [
            sample(0.31, (1.7, 0.0), coherence=0.9),
            sample(0.36, (1.7, 0.0), coherence=0.9),
        ],
    )
    assert events == []


def test_hud_arm_remaining_counts_down() -> None:
    d = detector(arm_window_s=3.0)
    arm(d)
    assert d.arm_remaining_s is not None
    assert abs(d.arm_remaining_s - 3.0) < 1e-9
    d.update(quiet(1.26))
    assert d.arm_remaining_s is not None
    assert abs(d.arm_remaining_s - 2.0) < 1e-9
