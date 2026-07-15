"""Pure gesture recognition: a stream of :class:`MotionState` -> discrete events.

All timing derives from ``MotionState.timestamp``, so the recognizer is fully
deterministic and testable with scripted streams. Stdlib only.
"""

from __future__ import annotations

from collections import deque
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


class SizeGrowSelectDetector:
    """SELECT strategy: blob area grows as the hand moves toward the camera.

    A push toward the camera enlarges the dominant motion blob while the
    centroid stays roughly in place. Fires when ``blob_area`` grows by at least
    ``area_growth`` relative to the oldest sample in a short history, then
    re-arms once area falls back near that baseline (hand pulled away).
    """

    def __init__(
        self,
        area_growth: float = 1.45,
        min_area: float = 0.002,
        steady_speed: float = 0.6,
        history: int = 5,
    ) -> None:
        self._area_growth = area_growth
        self._min_area = min_area
        self._steady_speed = steady_speed
        self._history_len = max(history, 2)
        self._history: deque[float] = deque(maxlen=self._history_len)
        self._armed = True
        self._baseline: float | None = None

    def update(self, state: MotionState) -> bool:
        """Rising-edge area growth detection; see class docstring."""
        area = state.blob_area
        self._history.append(area)

        if area < self._min_area:
            self._armed = True
            self._baseline = None
            return False

        if self._baseline is not None and area <= self._baseline * 1.1:
            self._armed = True
            self._baseline = None

        if not self._armed or len(self._history) < self._history_len:
            return False

        speed = _magnitude(state.velocity)
        if speed is not None and speed > self._steady_speed:
            return False

        baseline = self._history[0]
        if baseline < self._min_area:
            return False
        if area < baseline * self._area_growth:
            return False

        self._armed = False
        self._baseline = baseline
        return True


def _magnitude(vector: tuple[float, float] | None) -> float | None:
    if vector is None:
        return None
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5


_SWIPES = {
    ("x", 1): Event.SWIPE_RIGHT,
    ("x", -1): Event.SWIPE_LEFT,
    ("y", 1): Event.SWIPE_DOWN,
    ("y", -1): Event.SWIPE_UP,
}

_OPPOSITE = {
    Event.SWIPE_LEFT: Event.SWIPE_RIGHT,
    Event.SWIPE_RIGHT: Event.SWIPE_LEFT,
    Event.SWIPE_UP: Event.SWIPE_DOWN,
    Event.SWIPE_DOWN: Event.SWIPE_UP,
}


class GestureRecognizer:
    """Debounced state machine mapping motion states to gesture events.

    Swipes fire when the tracked centroid travels at least ``swipe_min_travel``
    (33% of the frame by default) along a clearly dominant axis. A low
    ``swipe_intent_speed`` is required only to *begin* a stroke; completion is
    travel-based so jump-capped tracking can still navigate. Strokes time out
    after ``swipe_max_duration_s``. The machine disarms until cursor speed drops
    below the release level after a swipe fires. Swipe and select use separate
    cooldowns. An opposite-direction lockout suppresses the return stroke. A
    second left swipe within ``double_swipe_window_s`` becomes
    DOUBLE_SWIPE_LEFT (back).

    The smoothed centroid is exposed as :attr:`cursor` for analog control.
    """

    def __init__(
        self,
        swipe_intent_speed: float = 0.10,
        swipe_release_factor: float = 0.5,
        swipe_axis_dominance: float = 2.0,
        swipe_max_duration_s: float = 3.0,
        event_cooldown_s: float = 0.75,
        select_cooldown_s: float = 1.0,
        opposite_lockout_s: float = 1.0,
        double_swipe_window_s: float = 0.9,
        cursor_smoothing: float = 0.5,
        min_track_area: float = 0.06,
        swipe_min_travel: float = 0.33,
        select_steady_speed: float = 0.6,
        select_detector: SelectDetector | None = None,
    ) -> None:
        self._intent_speed = swipe_intent_speed
        self._release_speed = swipe_intent_speed * swipe_release_factor
        self._axis_dominance = swipe_axis_dominance
        self._swipe_max_duration = swipe_max_duration_s
        self._swipe_cooldown = event_cooldown_s
        self._select_cooldown = select_cooldown_s
        self._opposite_lockout = opposite_lockout_s
        self._double_window = double_swipe_window_s
        self._smoothing = cursor_smoothing
        self._min_track_area = min_track_area
        self._swipe_min_travel = swipe_min_travel
        self._select_steady_speed = select_steady_speed
        self._select = select_detector if select_detector is not None else SizeGrowSelectDetector()

        self._cursor: tuple[float, float] | None = None
        self._prev_cursor_time = float("-inf")
        self._cursor_speed: float | None = None
        self._cursor_delta: tuple[float, float] = (0.0, 0.0)
        self._swipe_armed = True
        self._swipe_origin: tuple[float, float] | None = None
        self._swipe_axis: tuple[str, int] | None = None
        self._swipe_start_time = float("-inf")
        self._swipe_travel = 0.0
        self._last_swipe_event_time = float("-inf")
        self._last_select_time = float("-inf")
        self._last_left_time = float("-inf")
        self._last_swipe: Event | None = None
        self._last_swipe_time = float("-inf")

    @property
    def cursor(self) -> tuple[float, float] | None:
        """Smoothed centroid in normalized coordinates (the analog cursor)."""
        return self._cursor

    @property
    def cursor_speed(self) -> float | None:
        """Cursor displacement speed in normalized units per second."""
        return self._cursor_speed

    @property
    def swipe_travel(self) -> float:
        """Progress along the active swipe axis (0 when idle)."""
        return self._swipe_travel

    @property
    def swipe_min_travel(self) -> float:
        """Normalized distance required to complete a swipe."""
        return self._swipe_min_travel

    def update(self, state: MotionState) -> list[Event]:
        """Consume one motion state and return the events it triggered."""
        self._track_cursor(state)
        events: list[Event] = []

        swipe = self._detect_swipe(state)
        if swipe is not None:
            events.append(swipe)

        if (
            self._is_trackable(state)
            and self._can_select(state)
            and self._select.update(state)
            and self._select_cooldown_passed(state.timestamp)
        ):
            events.append(Event.SELECT)
            self._last_select_time = state.timestamp

        return events

    def _is_trackable(self, state: MotionState) -> bool:
        """Hand is visible enough to drive the cursor and gestures."""
        return state.centroid is not None and state.blob_area >= self._min_track_area

    def _can_select(self, state: MotionState) -> bool:
        """Block select during swipe strokes or lateral cursor movement."""
        if self._swipe_origin is not None:
            return False
        if self._cursor_speed is not None and self._cursor_speed > self._select_steady_speed:
            return False
        speed = _magnitude(state.velocity)
        return speed is None or speed <= self._select_steady_speed

    def _track_cursor(self, state: MotionState) -> None:
        if not self._is_trackable(state):
            self._cursor = None
            self._cursor_speed = None
            self._cursor_delta = (0.0, 0.0)
            return

        assert state.centroid is not None
        if self._cursor is None:
            self._cursor = state.centroid
            self._prev_cursor_time = state.timestamp
            self._cursor_speed = None
            self._cursor_delta = (0.0, 0.0)
        else:
            a = self._smoothing
            old = self._cursor
            new_cursor = (
                a * state.centroid[0] + (1.0 - a) * old[0],
                a * state.centroid[1] + (1.0 - a) * old[1],
            )
            dt = state.timestamp - self._prev_cursor_time
            if dt > 0.0:
                dx = new_cursor[0] - old[0]
                dy = new_cursor[1] - old[1]
                self._cursor_delta = (dx, dy)
                self._cursor_speed = (dx * dx + dy * dy) ** 0.5 / dt
            self._prev_cursor_time = state.timestamp
            self._cursor = new_cursor

    def _detect_swipe(self, state: MotionState) -> Event | None:
        if not self._is_trackable(state) or state.centroid is None:
            self._swipe_travel = 0.0
            return None

        speed = self._cursor_speed

        if (
            self._swipe_origin is not None
            and state.timestamp - self._swipe_start_time > self._swipe_max_duration
        ):
            self._reset_swipe_stroke()

        if not self._swipe_armed:
            if speed is None or speed < self._release_speed:
                self._swipe_armed = True
            else:
                return None

        if speed is None or speed < self._release_speed:
            if self._swipe_origin is not None:
                self._reset_swipe_stroke()
            self._swipe_travel = 0.0
            return None

        if not self._swipe_cooldown_passed(state.timestamp):
            return None

        if self._swipe_origin is None:
            if speed < self._intent_speed:
                return None
            axis_sign = self._dominant_axis(state)
            if axis_sign is None:
                return None
            self._swipe_origin = state.centroid
            self._swipe_axis = axis_sign
            self._swipe_start_time = state.timestamp
            self._swipe_travel = 0.0
            return None

        assert self._swipe_axis is not None
        axis, sign = self._swipe_axis
        ox, oy = self._swipe_origin
        cx, cy = state.centroid
        travel = (cx - ox) * sign if axis == "x" else (cy - oy) * sign
        self._swipe_travel = max(travel, 0.0)

        if self._swipe_travel < self._swipe_min_travel:
            return None

        event = _SWIPES[(axis, sign)]
        if self._is_opposite_locked(event, state.timestamp):
            self._reset_swipe_stroke()
            return None

        self._reset_swipe_stroke()
        self._swipe_armed = False
        self._last_swipe_event_time = state.timestamp
        self._last_swipe = event
        self._last_swipe_time = state.timestamp

        if event is Event.SWIPE_LEFT:
            if state.timestamp - self._last_left_time <= self._double_window:
                self._last_left_time = float("-inf")
                return Event.DOUBLE_SWIPE_LEFT
            self._last_left_time = state.timestamp
        return event

    def _dominant_axis(self, state: MotionState) -> tuple[str, int] | None:
        if state.velocity is not None:
            vx, vy = state.velocity
            if abs(vx) >= 1e-9 or abs(vy) >= 1e-9:
                return self._axis_from_delta(vx, vy)
        return self._axis_from_delta(self._cursor_delta[0], self._cursor_delta[1])

    def _axis_from_delta(self, dx: float, dy: float) -> tuple[str, int] | None:
        ax, ay = abs(dx), abs(dy)
        if ax < 1e-9 and ay < 1e-9:
            return None
        if ax >= ay:
            if ax < self._axis_dominance * ay:
                return None
            return "x", 1 if dx > 0 else -1
        if ay < self._axis_dominance * ax:
            return None
        return "y", 1 if dy > 0 else -1

    def _reset_swipe_stroke(self) -> None:
        self._swipe_origin = None
        self._swipe_axis = None
        self._swipe_travel = 0.0

    def _is_opposite_locked(self, event: Event, timestamp: float) -> bool:
        if self._last_swipe is None:
            return False
        if timestamp - self._last_swipe_time >= self._opposite_lockout:
            return False
        return event is _OPPOSITE.get(self._last_swipe)

    def _swipe_cooldown_passed(self, timestamp: float) -> bool:
        return timestamp - self._last_swipe_event_time >= self._swipe_cooldown

    def _select_cooldown_passed(self, timestamp: float) -> bool:
        return timestamp - self._last_select_time >= self._select_cooldown
