"""Pure dithering: quantize a ``[0, 1]`` signal into a ``{0, 1}`` mask.

Two algorithms: ordered (Bayer) dithering, which is fully vectorized and fast
enough for real time, and Floyd-Steinberg error diffusion, which is sequential
by nature and kept as a quality option. NumPy + stdlib only.
"""

from __future__ import annotations

from typing import assert_never

import numpy as np
import numpy.typing as npt

from motioncon.config import DitherMode

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.uint8]


def bayer_matrix(power: int) -> FloatArray:
    """Return the ``2**power`` square Bayer index matrix, built recursively.

    Entries are the integers ``0 .. n*n - 1`` in the classic recursive layout:
    ``M(2n) = [[4M, 4M+2], [4M+3, 4M+1]]``.
    """
    if power < 1:
        msg = f"power must be >= 1, got {power}"
        raise ValueError(msg)
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    for _ in range(power - 1):
        matrix = np.block(
            [
                [4 * matrix, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
    return matrix.astype(np.float32)


_BAYER_8 = bayer_matrix(3)
_BAYER_8_THRESHOLD = (_BAYER_8 + 0.5) / _BAYER_8.size


def ordered_dither(signal: FloatArray) -> MaskArray:
    """Bayer 8x8 ordered dithering: threshold each pixel against a tiled map."""
    h, w = signal.shape
    th, tw = _BAYER_8_THRESHOLD.shape
    reps_y = -(-h // th)
    reps_x = -(-w // tw)
    tiled = np.tile(_BAYER_8_THRESHOLD, (reps_y, reps_x))[:h, :w]
    return (signal > tiled).astype(np.uint8)


def floyd_steinberg(signal: FloatArray) -> MaskArray:
    """Floyd-Steinberg error diffusion with the standard 7/16, 3/16, 5/16, 1/16 kernel."""
    work = signal.astype(np.float64).copy()
    h, w = work.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            mask[y, x] = int(new)
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * (3 / 16)
                work[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1 / 16)
    return mask


def dither(signal: FloatArray, mode: DitherMode) -> MaskArray:
    """Dispatch to the requested dithering algorithm."""
    if mode is DitherMode.BAYER:
        return ordered_dither(signal)
    if mode is DitherMode.FLOYD_STEINBERG:
        return floyd_steinberg(signal)
    assert_never(mode)
