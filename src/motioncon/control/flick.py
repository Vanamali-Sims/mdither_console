"""Pure burst→quiet recognition of coherent directional-flow bursts."""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Protocol

from motioncon.config import Event


class DirectionSample(Protocol):
    """Input seam shared by optical flow and future direction sources."""

    @property
    def mean_flow(self) -> tuple[float, float]: ...

    @property
    def coherence(self) -> float: ...

    @property
    def active_frac(self) -> float: ...

    @property
    def timestamp(self) -> float: ...


class FlickPhase(Enum):
    """Externally visible detector phase for diagnostics and the HUD."""

    EMPTY = auto()
    ENTRY = auto()
    SETTLE = auto()
    ARMED = auto()
    STROKE = auto()
    FIRED = auto()


_EVENTS = {
    ("x", -1): Event.SWIPE_LEFT,
    ("y", -1): Event.FLICK_UP,
    ("y", 1): Event.FLICK_DOWN,
}


class FlickDetector:
    """State machine mapping direction samples to the three supported events.

    Arming is a burst→quiet latch: activity in the gesture band enters ENTRY,
    then a quiet gap settles into ARMED for ``arm_window_s``. Presence is not
    required while settling or armed — stillness produces zero optical flow.
    """

    def __init__(
        self,
        *,
        presence_floor: float = 0.015,
        quiet_frac: float = 0.02,
        settle_s: float = 0.25,
        settle_mag: float = 0.35,
        arm_window_s: float = 3.0,
        coh_min: float = 0.60,
        coherence_collapse: float = 0.35,
        flick_mag: float = 0.75,
        axis_dominance: float = 1.8,
        impulse_thresh: float = 3.0,
        stroke_max_s: float = 0.50,
        refractory_s: float = 0.40,
        opp_lockout_s: float = 0.70,
    ) -> None:
        self._presence_floor = presence_floor
        self._quiet_frac = quiet_frac
        self._settle_s = settle_s
        self._settle_mag = settle_mag
        self._arm_window_s = arm_window_s
        self._coh_min = coh_min
        self._coherence_collapse = coherence_collapse
        self._flick_mag = flick_mag
        self._axis_dominance = axis_dominance
        self._impulse_thresh = impulse_thresh
        self._stroke_max_s = stroke_max_s
        self._refractory_s = refractory_s
        self._opp_lockout_s = opp_lockout_s

        self._phase = FlickPhase.EMPTY
        self._settle_since: float | None = None
        self._armed_since: float | None = None
        self._last_t = 0.0
        self._stroke_since = 0.0
        self._stroke_axis: str | None = None
        self._stroke_sign = 0
        self._impulse = 0.0
        self._fired_at = float("-inf")
        self._last_axis: str | None = None
        self._last_sign = 0
        self._last_event_at = float("-inf")

    @property
    def phase(self) -> FlickPhase:
        """Current detector phase."""
        return self._phase

    @property
    def ready(self) -> bool:
        """Whether the detector is armed for a stroke."""
        return self._phase is FlickPhase.ARMED

    @property
    def arm_remaining_s(self) -> float | None:
        """Seconds left in the armed window, or ``None`` when not armed."""
        if self._phase is not FlickPhase.ARMED or self._armed_since is None:
            return None
        return max(0.0, self._arm_window_s - (self._last_t - self._armed_since))

    @property
    def impulse(self) -> float:
        """Signed accumulated flow along the locked stroke axis."""
        return self._impulse

    @property
    def progress(self) -> float:
        """Stroke completion fraction, clamped to ``[0, 1]``."""
        return min(abs(self._impulse) / self._impulse_thresh, 1.0)

    def update(self, sample: DirectionSample) -> list[Event]:
        """Consume one sample and return at most one directional event."""
        self._last_t = sample.timestamp
        magnitude = _magnitude(sample.mean_flow)
        quiet = self._is_quiet(sample.active_frac, magnitude)

        if self._phase is FlickPhase.EMPTY:
            if sample.active_frac >= self._presence_floor:
                self._phase = FlickPhase.ENTRY
            return []

        if self._phase is FlickPhase.FIRED:
            if sample.timestamp - self._fired_at < self._refractory_s:
                return []
            self._restart_settle(sample.timestamp, quiet)
            return []

        if self._phase is FlickPhase.ENTRY:
            if quiet:
                self._phase = FlickPhase.SETTLE
                self._settle_since = sample.timestamp
            return []

        if self._phase is FlickPhase.SETTLE:
            if not quiet:
                self._phase = FlickPhase.ENTRY
                self._settle_since = None
                return []
            if self._settle_since is None:
                self._settle_since = sample.timestamp
            if sample.timestamp - self._settle_since >= self._settle_s:
                self._phase = FlickPhase.ARMED
                self._armed_since = sample.timestamp
                self._settle_since = None
            return []

        if self._phase is FlickPhase.ARMED:
            assert self._armed_since is not None
            if sample.timestamp - self._armed_since >= self._arm_window_s:
                self._to_empty()
                return []
            axis_sign = self._stroke_candidate(sample, magnitude)
            if axis_sign is None:
                return []
            axis, sign = axis_sign
            if self._opposite_is_locked(axis, sign, sample.timestamp):
                self._phase = FlickPhase.ENTRY
                self._armed_since = None
                return []
            self._phase = FlickPhase.STROKE
            self._armed_since = None
            self._stroke_since = sample.timestamp
            self._stroke_axis = axis
            self._stroke_sign = sign
            self._impulse = self._axis_value(sample.mean_flow, axis)
            return self._maybe_fire(sample.timestamp)

        if sample.timestamp - self._stroke_since > self._stroke_max_s:
            self._phase = FlickPhase.ENTRY
            self._clear_stroke()
            return []
        if sample.coherence < self._coherence_collapse:
            self._phase = FlickPhase.ENTRY
            self._clear_stroke()
            return []

        assert self._stroke_axis is not None
        self._impulse += self._axis_value(sample.mean_flow, self._stroke_axis)
        return self._maybe_fire(sample.timestamp)

    def _is_quiet(self, active_frac: float, magnitude: float) -> bool:
        return active_frac < self._quiet_frac and magnitude < self._settle_mag

    def _stroke_candidate(
        self, sample: DirectionSample, magnitude: float
    ) -> tuple[str, int] | None:
        if sample.coherence < self._coh_min or magnitude < self._flick_mag:
            return None
        x, y = sample.mean_flow
        ax, ay = abs(x), abs(y)
        if ax >= ay:
            if ax < self._axis_dominance * ay:
                return None
            return ("x", 1 if x > 0.0 else -1)
        if ay < self._axis_dominance * ax:
            return None
        return ("y", 1 if y > 0.0 else -1)

    def _maybe_fire(self, timestamp: float) -> list[Event]:
        assert self._stroke_axis is not None
        if self._impulse * self._stroke_sign < self._impulse_thresh:
            return []

        event = _EVENTS.get((self._stroke_axis, self._stroke_sign))
        if event is None:
            self._phase = FlickPhase.ENTRY
            self._clear_stroke()
            return []

        axis = self._stroke_axis
        sign = self._stroke_sign
        self._phase = FlickPhase.FIRED
        self._fired_at = timestamp
        self._last_axis = axis
        self._last_sign = sign
        self._last_event_at = timestamp
        return [event]

    def _opposite_is_locked(self, axis: str, sign: int, timestamp: float) -> bool:
        return (
            axis == self._last_axis
            and sign == -self._last_sign
            and timestamp - self._last_event_at < self._opp_lockout_s
        )

    def _restart_settle(self, timestamp: float, quiet: bool) -> None:
        self._clear_stroke()
        self._armed_since = None
        if quiet:
            self._phase = FlickPhase.SETTLE
            self._settle_since = timestamp
        else:
            self._phase = FlickPhase.ENTRY
            self._settle_since = None

    def _to_empty(self) -> None:
        self._phase = FlickPhase.EMPTY
        self._settle_since = None
        self._armed_since = None
        self._clear_stroke()

    def _clear_stroke(self) -> None:
        self._stroke_axis = None
        self._stroke_sign = 0
        self._stroke_since = 0.0
        self._impulse = 0.0

    @staticmethod
    def _axis_value(vector: tuple[float, float], axis: str) -> float:
        return vector[0] if axis == "x" else vector[1]

    def reset(self) -> None:
        """Reset transient state while retaining no cooldown history."""
        self._to_empty()
        self._last_t = 0.0
        self._fired_at = float("-inf")
        self._last_axis = None
        self._last_sign = 0
        self._last_event_at = float("-inf")


def _magnitude(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])
