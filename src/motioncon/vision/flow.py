"""Dense optical-flow direction statistics with no position tracking or I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FlowState:
    """One frame's global motion summary inside the configured gesture band."""

    mean_flow: tuple[float, float]
    coherence: float
    active_frac: float
    timestamp: float

    @property
    def magnitude(self) -> float:
        """Magnitude of the weighted mean flow vector in pixels per frame."""
        x, y = self.mean_flow
        return math.hypot(x, y)


def _resize_bilinear(gray: FloatArray, width: int, height: int) -> FloatArray:
    """Resize a 2-D array using NumPy bilinear interpolation."""
    src_h, src_w = gray.shape
    if (src_w, src_h) == (width, height):
        return gray.astype(np.float32, copy=False)

    ys = np.linspace(0.0, src_h - 1, height, dtype=np.float32)
    xs = np.linspace(0.0, src_w - 1, width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0).reshape(-1, 1)
    wx = (xs - x0).reshape(1, -1)

    top = gray[y0[:, None], x0] * (1.0 - wx) + gray[y0[:, None], x1] * wx
    bottom = gray[y1[:, None], x0] * (1.0 - wx) + gray[y1[:, None], x1] * wx
    resized: FloatArray = (top * (1.0 - wy) + bottom * wy).astype(np.float32)
    return resized


class FlowAnalyzer:
    """Convert grayscale frames into band-limited dense-flow summaries."""

    def __init__(
        self,
        *,
        width: int = 96,
        height: int = 72,
        mag_floor: float = 0.35,
        gesture_band: tuple[float, float] = (0.25, 0.85),
        ignore_bottom: float = 0.15,
    ) -> None:
        if width < 2 or height < 2:
            msg = "flow dimensions must be at least 2x2"
            raise ValueError(msg)
        band_top, band_bottom = gesture_band
        if not 0.0 <= band_top < band_bottom <= 1.0:
            msg = f"invalid gesture band: {gesture_band}"
            raise ValueError(msg)
        if not 0.0 <= ignore_bottom < 1.0:
            msg = f"ignore_bottom must be in [0, 1), got {ignore_bottom}"
            raise ValueError(msg)
        self._width = width
        self._height = height
        self._mag_floor = mag_floor
        self._band_top = band_top
        self._band_bottom = min(band_bottom, 1.0 - ignore_bottom)
        self._previous: FloatArray | None = None

    def analyze(self, gray: FloatArray, timestamp: float) -> FlowState:
        """Measure coherent flow against the previous grayscale frame."""
        if gray.ndim != 2:
            msg = f"expected a 2-D grayscale frame, got {gray.shape}"
            raise ValueError(msg)
        current = _resize_bilinear(gray, self._width, self._height)
        if float(current.max(initial=0.0)) <= 1.0:
            current = (current * 255.0).astype(np.float32)
        top, bottom = self._band_bounds()
        current[:top] = 0.0
        current[bottom:] = 0.0
        if self._previous is None:
            self._previous = current.copy()
            return FlowState((0.0, 0.0), 0.0, 0.0, timestamp)

        initial = np.zeros((self._height, self._width, 2), dtype=np.float32)
        flow = cast(
            FloatArray,
            cv2.calcOpticalFlowFarneback(
                self._previous,
                current,
                initial,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            ),
        )
        self._previous = current.copy()
        return self._summarize(flow, timestamp)

    def _summarize(self, flow: FloatArray, timestamp: float) -> FlowState:
        top, bottom = self._band_bounds()
        band = flow[top:bottom]
        if band.size == 0:
            return FlowState((0.0, 0.0), 0.0, 0.0, timestamp)

        magnitude = np.linalg.norm(band, axis=2)
        active = magnitude > self._mag_floor
        active_count = int(np.count_nonzero(active))
        active_frac = active_count / active.size
        if active_count == 0:
            return FlowState((0.0, 0.0), 0.0, 0.0, timestamp)

        vectors = band[active]
        weights = magnitude[active]
        weight_sum = float(weights.sum())
        weighted = (vectors * weights[:, None]).sum(axis=0) / weight_sum
        vector_sum = vectors.sum(axis=0)
        coherence = float(np.linalg.norm(vector_sum) / weight_sum)
        mean_flow = (float(weighted[0]), float(weighted[1]))
        return FlowState(mean_flow, min(max(coherence, 0.0), 1.0), active_frac, timestamp)

    def _band_bounds(self) -> tuple[int, int]:
        return (
            int(round(self._band_top * self._height)),
            int(round(self._band_bottom * self._height)),
        )

    def reset(self) -> None:
        """Forget the previous frame."""
        self._previous = None
