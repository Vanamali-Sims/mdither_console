"""Pure motion analysis: difference signal -> :class:`MotionState`.

Finds moving blobs via a coarse energy grid, scores them with spatial priors
(left/right hands over center head, keyboard penalized), and hard-locks the
chosen target for a short window so the cursor cannot teleport. NumPy + stdlib.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BlobCandidate:
    """One motion blob with its spatial score."""

    centroid: tuple[float, float]
    area: float
    score: float


@dataclass(frozen=True, slots=True)
class MotionState:
    """Snapshot of the motion signal for one frame.

    Coordinates are normalized to ``[0, 1]`` (x right, y down); velocity is in
    normalized units per second. ``centroid`` is ``None`` when no target is
    actively tracked. ``blob_area`` is always the top candidate's patch fill
    ratio, even when tracking has not engaged.
    """

    timestamp: float
    energy: float
    centroid: tuple[float, float] | None
    velocity: tuple[float, float] | None
    blob_area: float = 0.0
    track_locked: bool = False
    lock_remaining_s: float = 0.0
    top_score: float = 0.0

    @property
    def has_motion(self) -> bool:
        """Whether a tracked target is active this frame."""
        return self.centroid is not None


def score_candidate(
    x: float,
    y: float,
    area: float,
    speed: float,
    *,
    hand_zone_x: float = 0.38,
    keyboard_y: float = 0.82,
    min_track_area: float = 0.10,
    intent_speed: float = 1.5,
    center_weight: float = 0.35,
    keyboard_weight: float = 0.15,
    intent_boost: float = 1.2,
) -> float:
    """Spatial prior: prefer left/right hands, penalize keyboard and center head."""
    if y > keyboard_y:
        weight = keyboard_weight
    elif x < hand_zone_x or x > (1.0 - hand_zone_x):
        weight = 1.0
    elif area >= min_track_area and speed >= intent_speed:
        weight = intent_boost
    else:
        weight = center_weight
    return area * (1.0 + speed) * weight


def _block_sum(signal: FloatArray, cell: int) -> FloatArray:
    """Sum ``signal`` over non-overlapping ``cell x cell`` blocks (edges cropped)."""
    h, w = signal.shape
    hc, wc = h // cell, w // cell
    cropped = signal[: hc * cell, : wc * cell]
    summed: FloatArray = cropped.reshape(hc, cell, wc, cell).sum(axis=(1, 3))
    return summed


class MotionAnalyzer:
    """Stateful analyzer with spatially-weighted multi-blob tracking."""

    def __init__(
        self,
        threshold: float = 0.12,
        min_energy: float = 1e-4,
        cell_size: int = 16,
        window_cells: int = 3,
        velocity_window: int = 5,
        min_track_area: float = 0.10,
        track_search_radius: float = 0.35,
        track_max_miss_frames: int = 8,
        track_candidates: int = 4,
        hand_zone_x: float = 0.38,
        keyboard_y: float = 0.82,
        intent_speed: float = 1.5,
        lock_duration_s: float = 2.0,
        max_centroid_step: float = 0.08,
        switch_margin: float = 1.3,
    ) -> None:
        self._threshold = threshold
        self._min_energy = min_energy
        self._cell_size = cell_size
        self._window_cells = window_cells
        self._min_track_area = min_track_area
        self._track_search_radius = track_search_radius
        self._track_max_miss_frames = track_max_miss_frames
        self._track_candidates = max(track_candidates, 1)
        self._hand_zone_x = hand_zone_x
        self._keyboard_y = keyboard_y
        self._intent_speed = intent_speed
        self._lock_duration = lock_duration_s
        self._max_step = max_centroid_step
        self._switch_margin = switch_margin
        self._history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=velocity_window)
        self._lock_centroid: tuple[float, float] | None = None
        self._lock_until = 0.0
        self._lock_score = 0.0
        self._miss_count = 0
        self._last_timestamp = 0.0

    def analyze(self, signal: FloatArray, timestamp: float) -> MotionState:
        """Compute the motion state for one difference signal."""
        active = np.where(signal >= self._threshold, signal, 0.0).astype(np.float32)
        energy = float(active.mean())
        h, w = active.shape

        candidates = self._find_candidates(active, energy, h, w, timestamp)
        top = candidates[0] if candidates else None
        raw_area = top.area if top is not None else 0.0
        top_score = top.score if top is not None else 0.0

        centroid_out: tuple[float, float] | None = None
        tracked = False
        measured = False
        hard_locked = self._lock_centroid is not None and timestamp < self._lock_until

        if hard_locked:
            assert self._lock_centroid is not None
            local = self._local_candidate(active, energy, h, w, self._lock_centroid, timestamp)
            if local is not None and local.area >= self._min_track_area:
                centroid_out = self._clamp_step(local.centroid, self._lock_centroid)
                self._lock_centroid = centroid_out
                self._miss_count = 0
                self._touch_lock(timestamp)
                measured = True
            else:
                centroid_out = self._lock_centroid
                self._miss_count += 1
            tracked = True
        else:
            best = candidates[0] if candidates else None
            if best is not None and best.area >= self._min_track_area:
                should_switch = (
                    self._lock_centroid is None
                    or best.score >= self._lock_score * self._switch_margin
                )
                if should_switch:
                    self._acquire_lock(best, timestamp)
                else:
                    self._touch_lock(timestamp)
                assert self._lock_centroid is not None
                centroid_out = self._clamp_step(best.centroid, self._lock_centroid)
                self._lock_centroid = centroid_out
                tracked = True
                self._miss_count = 0
                measured = True
            elif self._lock_centroid is not None and self._miss_count < self._track_max_miss_frames:
                if self._miss_count == 0:
                    self._history.clear()
                centroid_out = self._lock_centroid
                tracked = True
                self._miss_count += 1
            else:
                self._miss_count += 1
                if self._miss_count >= self._track_max_miss_frames:
                    self._release_lock()

        if tracked and centroid_out is not None:
            if measured:
                self._history.append((timestamp, centroid_out))
            velocity = self._drift_velocity()
        else:
            self._history.clear()
            velocity = None
            centroid_out = None

        lock_remaining = (
            max(0.0, self._lock_until - timestamp) if self._lock_centroid is not None else 0.0
        )
        is_locked = self._lock_centroid is not None and timestamp < self._lock_until
        self._last_timestamp = timestamp

        return MotionState(
            timestamp=timestamp,
            energy=energy,
            centroid=centroid_out,
            velocity=velocity,
            blob_area=raw_area,
            track_locked=is_locked and centroid_out is not None,
            lock_remaining_s=lock_remaining,
            top_score=top_score,
        )

    def _acquire_lock(self, candidate: BlobCandidate, timestamp: float) -> None:
        self._lock_centroid = candidate.centroid
        self._lock_score = candidate.score
        self._miss_count = 0
        self._touch_lock(timestamp)

    def _touch_lock(self, timestamp: float) -> None:
        """Extend the spatial lock while measurements keep arriving."""
        self._lock_until = timestamp + self._lock_duration

    def _release_lock(self) -> None:
        self._lock_centroid = None
        self._lock_until = 0.0
        self._lock_score = 0.0
        self._miss_count = 0

    def _clamp_step(
        self, new: tuple[float, float], old: tuple[float, float]
    ) -> tuple[float, float]:
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= self._max_step or dist <= 0.0:
            return new
        scale = self._max_step / dist
        return (old[0] + dx * scale, old[1] + dy * scale)

    def _estimate_speed(self, centroid: tuple[float, float], timestamp: float) -> float:
        if self._lock_centroid is None or timestamp <= self._last_timestamp:
            return 0.0
        dt = timestamp - self._last_timestamp
        if dt <= 0.0:
            return 0.0
        dx = centroid[0] - self._lock_centroid[0]
        dy = centroid[1] - self._lock_centroid[1]
        return (dx * dx + dy * dy) ** 0.5 / dt

    def _find_candidates(
        self, active: FloatArray, energy: float, h: int, w: int, timestamp: float
    ) -> list[BlobCandidate]:
        blobs = self._blobs_from_region(active, energy, h, w, offset_r=0, offset_c=0)
        scored: list[BlobCandidate] = []
        for centroid, area in blobs:
            speed = self._estimate_speed(centroid, timestamp)
            s = score_candidate(
                centroid[0],
                centroid[1],
                area,
                speed,
                hand_zone_x=self._hand_zone_x,
                keyboard_y=self._keyboard_y,
                min_track_area=self._min_track_area,
                intent_speed=self._intent_speed,
            )
            scored.append(BlobCandidate(centroid=centroid, area=area, score=s))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    def _local_candidate(
        self,
        active: FloatArray,
        energy: float,
        h: int,
        w: int,
        search_centroid: tuple[float, float],
        timestamp: float,
    ) -> BlobCandidate | None:
        r0, c0, r1, c1 = self._search_bounds(h, w, search_centroid)
        region = active[r0:r1, c0:c1]
        if region.size == 0 or float(region.sum()) <= 0.0:
            return None
        blobs = self._blobs_from_region(
            region, energy, h, w, offset_r=r0, offset_c=c0, max_peaks=1
        )
        if not blobs:
            return None
        centroid, area = blobs[0]
        speed = self._estimate_speed(centroid, timestamp)
        s = score_candidate(
            centroid[0],
            centroid[1],
            area,
            speed,
            hand_zone_x=self._hand_zone_x,
            keyboard_y=self._keyboard_y,
            min_track_area=self._min_track_area,
            intent_speed=self._intent_speed,
        )
        return BlobCandidate(centroid=centroid, area=area, score=s)

    def _blobs_from_region(
        self,
        region: FloatArray,
        energy: float,
        h: int,
        w: int,
        offset_r: int = 0,
        offset_c: int = 0,
        max_peaks: int | None = None,
    ) -> list[tuple[tuple[float, float], float]]:
        if energy < self._min_energy:
            return []

        cell = min(self._cell_size, region.shape[0], region.shape[1])
        if cell < 1:
            return []
        grid = _block_sum(region, cell)
        if float(grid.max()) <= 0.0:
            return []

        k = max_peaks if max_peaks is not None else self._track_candidates
        flat = grid.ravel()
        k = min(k, flat.size)
        indices = np.argpartition(-flat, k - 1)[:k]
        indices = indices[np.argsort(-flat[indices])]

        blobs: list[tuple[tuple[float, float], float]] = []
        seen: set[tuple[int, int]] = set()
        half = self._window_cells // 2

        for idx in indices:
            if flat[idx] <= 0.0:
                break
            peak_row, peak_col = np.unravel_index(int(idx), grid.shape)
            if (peak_row, peak_col) in seen:
                continue
            seen.add((peak_row, peak_col))

            r0 = max(int(peak_row) - half, 0) * cell
            c0 = max(int(peak_col) - half, 0) * cell
            r1 = min((int(peak_row) + half + 1) * cell, region.shape[0])
            c1 = min((int(peak_col) + half + 1) * cell, region.shape[1])

            patch = region[r0:r1, c0:c1]
            mass = float(patch.sum())
            if mass <= 0.0:
                continue

            ys, xs = np.mgrid[r0:r1, c0:c1].astype(np.float32)
            cy = float((ys * patch).sum() / mass) + offset_r
            cx = float((xs * patch).sum() / mass) + offset_c
            area = float(np.count_nonzero(patch)) / float(patch.size)
            blobs.append(((cx / max(w - 1, 1), cy / max(h - 1, 1)), area))

        return blobs

    def _search_bounds(
        self, h: int, w: int, centroid: tuple[float, float]
    ) -> tuple[int, int, int, int]:
        cx = centroid[0] * max(w - 1, 1)
        cy = centroid[1] * max(h - 1, 1)
        rx = self._track_search_radius * w * 0.5
        ry = self._track_search_radius * h * 0.5
        r0 = max(int(cy - ry), 0)
        r1 = min(int(cy + ry) + 1, h)
        c0 = max(int(cx - rx), 0)
        c1 = min(int(cx + rx) + 1, w)
        return r0, c0, r1, c1

    def _drift_velocity(self) -> tuple[float, float] | None:
        if len(self._history) < 2:
            return None
        (t0, (x0, y0)), (t1, (x1, y1)) = self._history[0], self._history[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return None
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def reset(self) -> None:
        """Forget centroid history (e.g. after the camera restarts)."""
        self._history.clear()
        self._release_lock()
        self._last_timestamp = 0.0
