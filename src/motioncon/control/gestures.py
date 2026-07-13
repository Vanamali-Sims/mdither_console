"""Pure gesture recognition: a stream of :class:`MotionState` -> discrete events.

All timing derives from ``MotionState.timestamp``, so the recognizer is fully
deterministic and testable with scripted streams. Stdlib only.
"""

from __future__ import annotations

import math
from typing import Protocol

from motioncon.config import Event
from motioncon.vision.motion import MotionState


class SelectDetector(Protocol):
    """Swappable strategy that decides when a SELECT gesture occurred.

    Implementations are fed every motion state and return ``True`` on the
    single frame the selection fires. Later alternatives (dwell, pinch via
    landmarks, ...) plug in through this interface.
    """

    def update(self, state: MotionState) -> bool:
        """Consume one motion state; return ``True`` to fire SELECT."""
        ...


class PushSelectDetector:
    """Default SELECT strategy: a brief, strong energy spike near a steady cursor.

    A "push" toward the camera produces a sudden burst of change energy while
    the centroid stays roughly in place. Fires on the rising edge of the energy
    crossing ``energy_threshold`` and re-arms once energy falls back below it.
    """

    def __init__(self, energy_threshold: float = 0.18, steady_speed: float = 0.6) -> None:
        self._energy_threshold = energy_threshold
        self._steady_speed = steady_speed
        self._armed = True

    def update(self, state: MotionState) -> bool:
        """Rising-edge spike detection; see class docstring."""
        if state.energy < self._energy_threshold:
            self._armed = True
            return False
        if not self._armed:
            return False
        speed = _magnitude(state.velocity)
        if speed is not None and speed > self._steady_speed:
            return False
        self._armed = False
        return True


def _magnitude(vector: tuple[float, float] | None) -> float | None:
    if vector is None:
        return None
    return math.hypot(vector[0], vector[1])


_SWIPES = {
    ("x", 1): Event.SWIPE_RIGHT,
    ("x", -1): Event.SWIPE_LEFT,
    ("y", 1): Event.SWIPE_DOWN,
    ("y", -1): Event.SWIPE_UP,
}


class GestureRecognizer:
    """Debounced state machine mapping motion states to gesture events.

    Swipes fire when the centroid velocity exceeds ``swipe_velocity_threshold``
    along a dominant axis; the machine then disarms until the speed drops below
    the release level, so one physical swipe emits one event. A second left
    swipe within ``double_swipe_window_s`` becomes DOUBLE_SWIPE_LEFT (back)
    instead of a plain SWIPE_LEFT. A shared cooldown debounces all events.

    The smoothed centroid is exposed as :attr:`cursor` for analog control.
    """

    def __init__(
        self,
        swipe_velocity_threshold: float = 1.2,
        swipe_release_factor: float = 0.5,
        event_cooldown_s: float = 0.35,
        double_swipe_window_s: float = 0.9,
        cursor_smoothing: float = 0.5,
        select_detector: SelectDetector | None = None,
    ) -> None:
        self._swipe_threshold = swipe_velocity_threshold
        self._release_speed = swipe_velocity_threshold * swipe_release_factor
        self._cooldown = event_cooldown_s
        self._double_window = double_swipe_window_s
        self._smoothing = cursor_smoothing
        self._select = select_detector if select_detector is not None else PushSelectDetector()

        self._cursor: tuple[float, float] | None = None
        self._swipe_armed = True
        self._last_event_time = float("-inf")
        self._last_left_time = float("-inf")

    @property
    def cursor(self) -> tuple[float, float] | None:
        """Smoothed centroid in normalized coordinates (the analog cursor)."""
        return self._cursor

    def update(self, state: MotionState) -> list[Event]:
        """Consume one motion state and return the events it triggered."""
        self._track_cursor(state)
        events: list[Event] = []

        swipe = self._detect_swipe(state)
        if swipe is not None:
            events.append(swipe)

        if self._select.update(state) and self._cooldown_passed(state.timestamp):
            events.append(Event.SELECT)
            self._last_event_time = state.timestamp

        return events

    def _track_cursor(self, state: MotionState) -> None:
        if state.centroid is None:
            return
        if self._cursor is None:
            self._cursor = state.centroid
        else:
            a = self._smoothing
            self._cursor = (
                a * state.centroid[0] + (1.0 - a) * self._cursor[0],
                a * state.centroid[1] + (1.0 - a) * self._cursor[1],
            )

    def _detect_swipe(self, state: MotionState) -> Event | None:
        speed = _magnitude(state.velocity)
        if speed is None or speed < self._release_speed:
            self._swipe_armed = True
        if speed is None or speed < self._swipe_threshold:
            return None
        if not self._swipe_armed or not self._cooldown_passed(state.timestamp):
            return None

        assert state.velocity is not None
        vx, vy = state.velocity
        axis, sign = (
            ("x", 1 if vx > 0 else -1) if abs(vx) >= abs(vy) else ("y", 1 if vy > 0 else -1)
        )
        event = _SWIPES[(axis, sign)]

        self._swipe_armed = False
        self._last_event_time = state.timestamp

        if event is Event.SWIPE_LEFT:
            if state.timestamp - self._last_left_time <= self._double_window:
                self._last_left_time = float("-inf")
                return Event.DOUBLE_SWIPE_LEFT
            self._last_left_time = state.timestamp
        return event

    def _cooldown_passed(self, timestamp: float) -> bool:
        return timestamp - self._last_event_time >= self._cooldown
