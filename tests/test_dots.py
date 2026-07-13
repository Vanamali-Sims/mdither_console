"""Tests for the pure dot-field rasterizer."""

from __future__ import annotations

import numpy as np
import pytest

from motioncon.render.dots import downsample, render_dots

FG = (40, 255, 120)
BG = (0, 0, 0)


class TestDownsample:
    def test_block_means(self) -> None:
        signal = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        out = downsample(signal, cell_size=2)
        np.testing.assert_allclose(out, [[1.0, 0.0], [0.0, 0.5]], atol=1e-6)

    def test_crops_partial_edge_cells(self) -> None:
        signal = np.ones((5, 7), dtype=np.float32)
        assert downsample(signal, cell_size=2).shape == (2, 3)

    def test_rejects_too_small_signal(self) -> None:
        with pytest.raises(ValueError, match="smaller"):
            downsample(np.ones((4, 4), dtype=np.float32), cell_size=8)


class TestRenderDots:
    def test_canvas_shape_and_dtype(self) -> None:
        mask = np.zeros((3, 5), dtype=np.uint8)
        canvas = render_dots(mask, cell_size=8, dot_radius=0.4, foreground=FG, background=BG)
        assert canvas.shape == (24, 40, 3)
        assert canvas.dtype == np.uint8

    def test_empty_mask_is_all_background(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        canvas = render_dots(mask, cell_size=8, dot_radius=0.4, foreground=FG, background=(9, 8, 7))
        assert set(np.unique(canvas[:, :, 0]).tolist()) == {9}
        assert set(np.unique(canvas[:, :, 1]).tolist()) == {8}
        assert set(np.unique(canvas[:, :, 2]).tolist()) == {7}

    def test_dot_appears_only_in_set_cell(self) -> None:
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 1] = 1
        canvas = render_dots(mask, cell_size=8, dot_radius=0.4, foreground=FG, background=BG)

        set_cell = canvas[0:8, 8:16]
        assert int(set_cell.sum()) > 0
        for r0, c0 in ((0, 0), (8, 0), (8, 8)):
            assert int(canvas[r0 : r0 + 8, c0 : c0 + 8].sum()) == 0

    def test_dot_center_is_full_foreground(self) -> None:
        mask = np.ones((1, 1), dtype=np.uint8)
        canvas = render_dots(mask, cell_size=9, dot_radius=0.4, foreground=FG, background=BG)
        np.testing.assert_array_equal(canvas[4, 4], FG)

    def test_dot_corners_stay_background(self) -> None:
        mask = np.ones((1, 1), dtype=np.uint8)
        canvas = render_dots(mask, cell_size=8, dot_radius=0.35, foreground=FG, background=BG)
        np.testing.assert_array_equal(canvas[0, 0], BG)
        np.testing.assert_array_equal(canvas[-1, -1], BG)

    def test_brightness_scales_dot_intensity(self) -> None:
        mask = np.ones((1, 1), dtype=np.uint8)
        full = render_dots(mask, cell_size=9, dot_radius=0.4, foreground=FG, background=BG)
        half = render_dots(
            mask,
            cell_size=9,
            dot_radius=0.4,
            foreground=FG,
            background=BG,
            brightness=np.full((1, 1), 0.5, dtype=np.float32),
        )
        np.testing.assert_allclose(
            half[4, 4].astype(np.float32), full[4, 4].astype(np.float32) * 0.5, atol=1.0
        )

    def test_brightness_shape_mismatch_raises(self) -> None:
        mask = np.ones((2, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="brightness"):
            render_dots(
                mask,
                cell_size=4,
                dot_radius=0.4,
                foreground=FG,
                background=BG,
                brightness=np.ones((3, 3), dtype=np.float32),
            )

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(5)
        mask = (rng.random((6, 6)) > 0.5).astype(np.uint8)
        a = render_dots(mask, cell_size=6, dot_radius=0.4, foreground=FG, background=BG)
        b = render_dots(mask, cell_size=6, dot_radius=0.4, foreground=FG, background=BG)
        np.testing.assert_array_equal(a, b)
