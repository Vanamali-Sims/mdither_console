"""Burst→quiet arming with capture→classify horizontal stroke detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
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
    CAPTURE = auto()
    FIRED = auto()


@dataclass(frozen=True, slots=True)
class BurstSegment:
    """One consistent-sign run inside a captured burst."""

    sign: int
    integral: float
    duration: float
    mean_coherence: float


@dataclass(frozen=True, slots=True)
class BurstReport:
    """Finished burst ready for telemetry / offline tuning."""

    start_ts: float
    end_ts: float
    n_samples: int
    segments: tuple[BurstSegment, ...]
    outcome: str
    y_integral: float


@dataclass(frozen=True, slots=True)
class _CaptureSample:
    timestamp: float
    mean_flow: tuple[float, float]
    coherence: float
    active_frac: float
    magnitude: float


_ZERO_EPS = 1e-12
_MAX_DEAD_IN_SEGMENT = 2
_REENTRY_FRAMES = 2


class FlickDetector:
    """Arm with burst→quiet, then capture→classify horizontal throws.

    Presence is not required while settling or armed. Settling uses a rolling
    quiet-fraction window so single tremor blips do not reset the timer.
    Capture keeps dead frames mid-burst (flow blink is normal). Classification
    ignores vertical motion and fires the first time-ordered segment that clears
    the throw threshold.
    """

    def __init__(
        self,
        *,
        presence_floor: float = 0.015,
        quiet_frac: float = 0.02,
        settle_s: float = 0.25,
        settle_mag: float = 0.35,
        settle_quiet_frac: float = 0.75,
        arm_window_s: float = 3.0,
        capture_floor: float = 0.15,
        reentry_mag: float | None = None,
        burst_quiet_s: float = 0.20,
        burst_max_s: float = 1.0,
        throw_impulse: float = 0.05,
        coh_min: float = 0.50,
        refractory_s: float = 0.40,
    ) -> None:
        self._presence_floor = presence_floor
        self._quiet_frac = quiet_frac
        self._settle_s = settle_s
        self._settle_mag = settle_mag
        self._settle_quiet_frac = settle_quiet_frac
        self._arm_window_s = arm_window_s
        self._capture_floor = capture_floor
        self._reentry_mag = capture_floor if reentry_mag is None else reentry_mag
        self._burst_quiet_s = burst_quiet_s
        self._burst_max_s = burst_max_s
        self._throw_impulse = throw_impulse
        self._coh_min = coh_min
        self._refractory_s = refractory_s

        self._phase = FlickPhase.EMPTY
        self._settle_window: deque[tuple[float, bool]] = deque()
        self._reentry_run = 0
        self._armed_since: float | None = None
        self._last_t = 0.0
        self._fired_at = float("-inf")

        self._buffer: list[_CaptureSample] = []
        self._capture_started: float | None = None
        self._quiet_since: float | None = None
        self._left_energy = 0.0
        self._right_energy = 0.0

        self._pending_burst: BurstReport | None = None
        self._feedback: str | None = None
        self._feedback_at = float("-inf")

    @property
    def phase(self) -> FlickPhase:
        """Current detector phase."""
        return self._phase

    @property
    def ready(self) -> bool:
        """Whether the detector is armed and waiting to open a capture."""
        return self._phase is FlickPhase.ARMED

    @property
    def capturing(self) -> bool:
        """Whether a burst buffer is currently open."""
        return self._phase is FlickPhase.CAPTURE

    @property
    def arm_remaining_s(self) -> float | None:
        """Seconds left in the armed window, or ``None`` when not armed."""
        if self._phase is not FlickPhase.ARMED or self._armed_since is None:
            return None
        return max(0.0, self._arm_window_s - (self._last_t - self._armed_since))

    @property
    def capture_balance(self) -> tuple[float, float] | None:
        """``(left_frac, right_frac)`` of accumulated ``|sx|``, or ``None``."""
        if self._phase is not FlickPhase.CAPTURE:
            return None
        total = self._left_energy + self._right_energy
        if total <= _ZERO_EPS:
            return (0.5, 0.5)
        return (self._left_energy / total, self._right_energy / total)

    @property
    def feedback(self) -> str | None:
        """Last classify label: ``STROKE_LEFT``, ``STROKE_RIGHT``, or ``?``."""
        return self._feedback

    @property
    def feedback_at(self) -> float:
        """Timestamp when classify feedback was last set."""
        return self._feedback_at

    def take_burst(self) -> BurstReport | None:
        """Return and clear the latest finished burst report, if any."""
        report = self._pending_burst
        self._pending_burst = None
        return report

    def update(self, sample: DirectionSample) -> list[Event]:
        """Consume one sample and return at most one directional event."""
        self._last_t = sample.timestamp
        magnitude = _magnitude(sample.mean_flow)
        quiet = self._is_settle_quiet(magnitude)

        if self._phase is FlickPhase.EMPTY:
            if sample.active_frac >= self._presence_floor:
                self._phase = FlickPhase.ENTRY
            return []

        if self._phase is FlickPhase.FIRED:
            if sample.timestamp - self._fired_at < self._refractory_s:
                return []
            self._restart_settle(sample.timestamp, quiet, magnitude)
            return []

        if self._phase is FlickPhase.ENTRY:
            if quiet:
                self._phase = FlickPhase.SETTLE
                self._begin_settle(sample.timestamp, quiet)
            return []

        if self._phase is FlickPhase.SETTLE:
            if self._note_reentry(magnitude):
                self._phase = FlickPhase.ENTRY
                self._clear_settle()
                return []
            if self._note_settle_sample(sample.timestamp, quiet):
                self._phase = FlickPhase.ARMED
                self._armed_since = sample.timestamp
                self._clear_settle()
            return []

        if self._phase is FlickPhase.ARMED:
            assert self._armed_since is not None
            if sample.timestamp - self._armed_since >= self._arm_window_s:
                self._to_empty()
                return []
            if magnitude >= self._capture_floor:
                self._open_capture(sample, magnitude)
            return []

        assert self._phase is FlickPhase.CAPTURE
        return self._continue_capture(sample, magnitude)

    def _is_settle_quiet(self, magnitude: float) -> bool:
        return magnitude < self._settle_mag

    def _begin_settle(self, timestamp: float, quiet: bool) -> None:
        self._settle_window.clear()
        self._reentry_run = 0
        self._settle_window.append((timestamp, quiet))

    def _clear_settle(self) -> None:
        self._settle_window.clear()
        self._reentry_run = 0

    def _note_reentry(self, magnitude: float) -> bool:
        """Return True when sustained motion should demote SETTLE → ENTRY."""
        if magnitude >= self._reentry_mag:
            self._reentry_run += 1
        else:
            self._reentry_run = 0
        return self._reentry_run >= _REENTRY_FRAMES

    def _note_settle_sample(self, timestamp: float, quiet: bool) -> bool:
        """Append one settle sample; return True when the rolling window arms."""
        self._settle_window.append((timestamp, quiet))
        # Keep a rolling window of about settle_s without shrinking span below it.
        while (
            len(self._settle_window) >= 2
            and self._settle_window[-1][0] - self._settle_window[1][0] >= self._settle_s
        ):
            self._settle_window.popleft()

        span = self._settle_window[-1][0] - self._settle_window[0][0]
        if span < self._settle_s:
            return False
        quiet_count = sum(1 for _, is_quiet in self._settle_window if is_quiet)
        return quiet_count / len(self._settle_window) >= self._settle_quiet_frac

    def _open_capture(self, sample: DirectionSample, magnitude: float) -> None:
        self._phase = FlickPhase.CAPTURE
        # Keep _armed_since so an ambiguous classify can return to ARMED.
        self._capture_started = sample.timestamp
        self._quiet_since = None
        self._buffer = []
        self._left_energy = 0.0
        self._right_energy = 0.0
        self._append_capture(sample, magnitude)

    def _continue_capture(self, sample: DirectionSample, magnitude: float) -> list[Event]:
        self._append_capture(sample, magnitude)
        assert self._capture_started is not None

        if magnitude < self._capture_floor:
            if self._quiet_since is None:
                self._quiet_since = sample.timestamp
        else:
            self._quiet_since = None

        quiet_done = (
            self._quiet_since is not None
            and sample.timestamp - self._quiet_since >= self._burst_quiet_s
        )
        max_done = sample.timestamp - self._capture_started >= self._burst_max_s
        if quiet_done or max_done:
            return self._finish_capture(sample.timestamp)
        return []

    def _append_capture(self, sample: DirectionSample, magnitude: float) -> None:
        self._buffer.append(
            _CaptureSample(
                timestamp=sample.timestamp,
                mean_flow=sample.mean_flow,
                coherence=sample.coherence,
                active_frac=sample.active_frac,
                magnitude=magnitude,
            )
        )
        sx = sample.mean_flow[0] * sample.active_frac
        if sx < 0.0:
            self._left_energy += abs(sx)
        elif sx > 0.0:
            self._right_energy += abs(sx)

    def _finish_capture(self, end_ts: float) -> list[Event]:
        report, event = _classify_burst(
            self._buffer,
            throw_impulse=self._throw_impulse,
            coh_min=self._coh_min,
        )
        self._pending_burst = report
        self._feedback = report.outcome if event is None else event.name
        if self._feedback == "none":
            self._feedback = "?"
        self._feedback_at = end_ts
        self._clear_capture()

        if event is not None:
            self._phase = FlickPhase.FIRED
            self._fired_at = end_ts
            self._armed_since = None
            return [event]

        # No throw: stay armed if the window remains, else EMPTY.
        if self._armed_since is not None and end_ts - self._armed_since < self._arm_window_s:
            self._phase = FlickPhase.ARMED
        else:
            self._to_empty()
        return []

    def _restart_settle(self, timestamp: float, quiet: bool, magnitude: float) -> None:
        self._clear_capture()
        self._armed_since = None
        if quiet:
            self._phase = FlickPhase.SETTLE
            self._begin_settle(timestamp, quiet)
            # A loud first sample after refractory still counts toward reentry.
            self._note_reentry(magnitude)
        else:
            self._phase = FlickPhase.ENTRY
            self._clear_settle()

    def _to_empty(self) -> None:
        self._phase = FlickPhase.EMPTY
        self._clear_settle()
        self._armed_since = None
        self._clear_capture()

    def _clear_capture(self) -> None:
        self._buffer = []
        self._capture_started = None
        self._quiet_since = None
        self._left_energy = 0.0
        self._right_energy = 0.0

    def reset(self) -> None:
        """Reset transient state while retaining no cooldown history."""
        self._to_empty()
        self._last_t = 0.0
        self._fired_at = float("-inf")
        self._pending_burst = None
        self._feedback = None
        self._feedback_at = float("-inf")


def _magnitude(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _sx(sample: _CaptureSample) -> float:
    return sample.mean_flow[0] * sample.active_frac


def _sign_sx(sx: float) -> int:
    if abs(sx) <= _ZERO_EPS:
        return 0
    return 1 if sx > 0.0 else -1


def _segment_runs(samples: list[_CaptureSample]) -> list[list[_CaptureSample]]:
    """Split into consistent-sign runs; tolerate up to two consecutive dead samples."""
    segments: list[list[_CaptureSample]] = []
    current: list[_CaptureSample] = []
    current_sign = 0
    zero_run = 0

    def flush() -> None:
        nonlocal current, current_sign, zero_run
        if current:
            # Drop trailing dead samples from the segment body.
            while current and _sign_sx(_sx(current[-1])) == 0:
                current.pop()
            if current:
                segments.append(current)
        current = []
        current_sign = 0
        zero_run = 0

    for sample in samples:
        sx = _sx(sample)
        sign = _sign_sx(sx)
        if sign == 0:
            if current_sign == 0:
                continue
            zero_run += 1
            if zero_run > _MAX_DEAD_IN_SEGMENT:
                flush()
            else:
                current.append(sample)
            continue

        zero_run = 0
        if current_sign == 0:
            current_sign = sign
            current = [sample]
        elif sign == current_sign:
            current.append(sample)
        else:
            flush()
            current_sign = sign
            current = [sample]

    flush()
    return segments


def _integrate_segment(samples: list[_CaptureSample]) -> BurstSegment:
    """Time-integrate weighted horizontal flow and coherence over one segment."""
    assert samples
    sign = 0
    for sample in samples:
        sign = _sign_sx(_sx(sample))
        if sign != 0:
            break
    assert sign != 0

    integral = 0.0
    coh_weight = 0.0
    coh_sum = 0.0
    for index, sample in enumerate(samples):
        if index == 0:
            dt = samples[1].timestamp - sample.timestamp if len(samples) >= 2 else 1.0 / 15.0
        else:
            dt = sample.timestamp - samples[index - 1].timestamp
        dt = max(dt, 0.0)
        sx = _sx(sample)
        integral += sx * dt
        weight = max(sample.active_frac, 0.0)
        coh_sum += sample.coherence * weight
        coh_weight += weight

    duration = samples[-1].timestamp - samples[0].timestamp
    if len(samples) == 1:
        duration = max(duration, 1.0 / 15.0)
    mean_coh = coh_sum / coh_weight if coh_weight > _ZERO_EPS else 0.0
    return BurstSegment(
        sign=sign,
        integral=integral,
        duration=duration,
        mean_coherence=mean_coh,
    )


def _classify_burst(
    samples: list[_CaptureSample],
    *,
    throw_impulse: float,
    coh_min: float,
) -> tuple[BurstReport, Event | None]:
    if not samples:
        report = BurstReport(
            start_ts=0.0,
            end_ts=0.0,
            n_samples=0,
            segments=(),
            outcome="none",
            y_integral=0.0,
        )
        return report, None

    y_integral = 0.0
    for index, sample in enumerate(samples):
        if index == 0:
            dt = (samples[1].timestamp - sample.timestamp) if len(samples) >= 2 else 1.0 / 15.0
        else:
            dt = sample.timestamp - samples[index - 1].timestamp
        y_integral += sample.mean_flow[1] * sample.active_frac * max(dt, 0.0)

    segments = tuple(_integrate_segment(run) for run in _segment_runs(samples))
    event: Event | None = None
    for segment in segments:
        if abs(segment.integral) >= throw_impulse and segment.mean_coherence >= coh_min:
            event = Event.STROKE_RIGHT if segment.sign > 0 else Event.STROKE_LEFT
            break

    report = BurstReport(
        start_ts=samples[0].timestamp,
        end_ts=samples[-1].timestamp,
        n_samples=len(samples),
        segments=segments,
        outcome=event.name if event is not None else "none",
        y_integral=y_integral,
    )
    return report, event
