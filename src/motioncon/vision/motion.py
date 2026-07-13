"""Pure motion analysis: difference signal -> :class:`MotionState`.

Finds the dominant moving blob via a coarse energy grid, computes its
intensity-weighted centroid, the total change energy, and the centroid's
velocity over a short window of recent frames. NumPy + stdlib only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MotionState:
    """Snapshot of the motion signal for one frame.

    Coordinates are normalized to ``[0, 1]`` (x right, y down); velocity is in
    normalized units per second. ``centroid`` and ``velocity`` are ``None``
    when no motion is detected.
    """

    timestamp: float
    energy: float
    centroid: tuple[float, float] | None
    velocity: tuple[float, float] | None

    @property
    def has_motion(self) -> bool:
        """Whether a dominant moving blob was found this frame."""
        return self.centroid is not None


def _block_sum(signal: FloatArray, cell: int) -> FloatArray:
    """Sum ``signal`` over non-overlapping ``cell x cell`` blocks (edges cropped)."""
    h, w = signal.shape
    hc, wc = h // cell, w // cell
    cropped = signal[: hc * cell, : wc * cell]
    summed: FloatArray = cropped.reshape(hc, cell, wc, cell).sum(axis=(1, 3))
    return summed


class MotionAnalyzer:
    """Stateful analyzer turning difference signals into :class:`MotionState`.

    The dominant blob is located by finding the strongest cell of a coarse
    energy grid and taking the weighted centroid of the thresholded signal in a
    small window of cells around it, which rejects isolated noise elsewhere in
    the frame. Velocity is the centroid drift across a deque of recent frames.
    """

    def __init__(
        self,
        threshold: float = 0.12,
        min_energy: float = 1e-4,
        cell_size: int = 16,
        window_cells: int = 3,
        velocity_window: int = 5,
    ) -> None:
        self._threshold = threshold
        self._min_energy = min_energy
        self._cell_size = cell_size
        self._window_cells = window_cells
        self._history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=velocity_window)

    def analyze(self, signal: FloatArray, timestamp: float) -> MotionState:
        """Compute the motion state for one difference signal.

        ``signal`` must be a 2-D float array in ``[0, 1]``; ``timestamp`` is in
        seconds (any monotonic clock).
        """
        active = np.where(signal >= self._threshold, signal, 0.0).astype(np.float32)
        energy = float(active.mean())

        centroid = self._dominant_centroid(active, energy)
        if centroid is None:
            self._history.clear()
            return MotionState(timestamp=timestamp, energy=energy, centroid=None, velocity=None)

        self._history.append((timestamp, centroid))
        return MotionState(
            timestamp=timestamp,
            energy=energy,
            centroid=centroid,
            velocity=self._drift_velocity(),
        )

    def _dominant_centroid(self, active: FloatArray, energy: float) -> tuple[float, float] | None:
        if energy < self._min_energy:
            return None

        h, w = active.shape
        cell = min(self._cell_size, h, w)
        grid = _block_sum(active, cell)
        peak_row, peak_col = np.unravel_index(int(grid.argmax()), grid.shape)

        # Window of cells around the peak, in pixel coordinates.
        half = self._window_cells // 2
        r0 = max(int(peak_row) - half, 0) * cell
        c0 = max(int(peak_col) - half, 0) * cell
        r1 = min((int(peak_row) + half + 1) * cell, h)
        c1 = min((int(peak_col) + half + 1) * cell, w)

        patch = active[r0:r1, c0:c1]
        mass = float(patch.sum())
        if mass <= 0.0:
            return None

        ys, xs = np.mgrid[r0:r1, c0:c1].astype(np.float32)
        cy = float((ys * patch).sum() / mass)
        cx = float((xs * patch).sum() / mass)
        return (cx / max(w - 1, 1), cy / max(h - 1, 1))

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
