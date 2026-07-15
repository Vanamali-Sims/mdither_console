"""Pure frame-differencing primitives.

Only NumPy and the stdlib: no OpenCV, no I/O. All signals are ``float32``
arrays with values in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def to_gray(frame: npt.NDArray[np.uint8]) -> FloatArray:
    """Convert an ``(H, W, 3)`` uint8 frame to ``(H, W)`` float32 luminance in [0, 1].

    A 2-D input is treated as already grayscale and only rescaled.
    """
    scaled = frame.astype(np.float32) / 255.0
    if scaled.ndim == 2:
        return scaled
    if scaled.ndim == 3 and scaled.shape[2] == 3:
        # ITU-R BT.601 luma weights.
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray: FloatArray = (scaled @ weights).astype(np.float32)
        return gray
    msg = f"expected (H, W) or (H, W, 3) frame, got shape {frame.shape}"
    raise ValueError(msg)


def frame_difference(gray: FloatArray, prev: FloatArray, noise_floor: float = 0.0) -> FloatArray:
    """Absolute temporal difference of two grayscale frames, clipped to [0, 1].

    Values below ``noise_floor`` are zeroed to suppress sensor noise.
    """
    if gray.shape != prev.shape:
        msg = f"frame shapes differ: {gray.shape} vs {prev.shape}"
        raise ValueError(msg)
    diff: FloatArray = np.clip(np.abs(gray - prev), 0.0, 1.0).astype(np.float32)
    if noise_floor > 0.0:
        diff = np.where(diff >= noise_floor, diff, 0.0).astype(np.float32)
    return diff


def boost(signal: FloatArray, gain: float) -> FloatArray:
    """Amplify a weak signal by ``gain`` and clip back to [0, 1]."""
    boosted: FloatArray = np.clip(signal * gain, 0.0, 1.0).astype(np.float32)
    return boosted


class TrailsAccumulator:
    """Decaying motion-trails buffer: ``accum = max(signal, accum * decay)``.

    The accumulator brightens instantly where motion occurs and fades
    exponentially where it stops, leaving ghost trails behind moving objects.
    """

    def __init__(self, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            msg = f"decay must be in [0, 1], got {decay}"
            raise ValueError(msg)
        self._decay = decay
        self._accum: FloatArray | None = None

    @property
    def value(self) -> FloatArray | None:
        """Current accumulator contents, or ``None`` before the first update."""
        return self._accum

    def update(self, signal: FloatArray) -> FloatArray:
        """Fold a new difference signal into the trails buffer and return it."""
        if self._accum is None or self._accum.shape != signal.shape:
            self._accum = signal.astype(np.float32).copy()
        else:
            self._accum = np.maximum(signal, self._accum * self._decay).astype(np.float32)
        return self._accum

    def reset(self) -> None:
        """Clear the buffer; the next update starts from the raw signal."""
        self._accum = None
