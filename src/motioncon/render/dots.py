"""Pure dot-field rasterizer: dither mask -> colored dot canvas.

Fully vectorized: the canvas is assembled with broadcasting only, no Python
loops over pixels. NumPy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.uint8]
ImageArray = npt.NDArray[np.uint8]


def downsample(signal: FloatArray, cell_size: int) -> FloatArray:
    """Block-mean the signal so each ``cell_size`` square becomes one dot cell.

    Edge rows/columns that don't fill a whole cell are cropped.
    """
    if cell_size < 1:
        msg = f"cell_size must be >= 1, got {cell_size}"
        raise ValueError(msg)
    h, w = signal.shape
    hc, wc = h // cell_size, w // cell_size
    if hc == 0 or wc == 0:
        msg = f"signal {signal.shape} smaller than one {cell_size}px cell"
        raise ValueError(msg)
    cropped = signal[: hc * cell_size, : wc * cell_size]
    result: FloatArray = (
        cropped.reshape(hc, cell_size, wc, cell_size).mean(axis=(1, 3)).astype(np.float32)
    )
    return result


def _disk(cell_size: int, radius_fraction: float) -> FloatArray:
    """Anti-aliased disk template of shape ``(cell_size, cell_size)`` in [0, 1]."""
    center = (cell_size - 1) / 2.0
    radius = radius_fraction * cell_size
    yy, xx = np.mgrid[:cell_size, :cell_size].astype(np.float32)
    dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    disk: FloatArray = np.clip(radius + 0.5 - dist, 0.0, 1.0).astype(np.float32)
    return disk


def render_dots(
    mask: MaskArray,
    cell_size: int,
    dot_radius: float,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    brightness: FloatArray | None = None,
) -> ImageArray:
    """Rasterize a ``(H, W)`` mask into an ``(H*cell, W*cell, 3)`` uint8 RGB image.

    Cells where ``mask`` is 1 get a foreground-colored dot; everything else is
    background. ``brightness`` (same shape as ``mask``, values in [0, 1])
    optionally scales each dot's intensity so trails fade visually.
    """
    h, w = mask.shape
    intensity = mask.astype(np.float32)
    if brightness is not None:
        if brightness.shape != mask.shape:
            msg = f"brightness shape {brightness.shape} != mask shape {mask.shape}"
            raise ValueError(msg)
        intensity = intensity * np.clip(brightness, 0.0, 1.0)

    disk = _disk(cell_size, dot_radius)
    # (H, W, cell, cell) -> (H*cell, W*cell) coverage field in [0, 1].
    field = (intensity[:, :, None, None] * disk[None, None, :, :]).transpose(0, 2, 1, 3)
    coverage = field.reshape(h * cell_size, w * cell_size)

    fg = np.array(foreground, dtype=np.float32)
    bg = np.array(background, dtype=np.float32)
    canvas = bg[None, None, :] + coverage[:, :, None] * (fg - bg)[None, None, :]
    result: ImageArray = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    return result
