"""Tests for pure dithering algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from motioncon.config import DitherMode
from motioncon.render.dither import bayer_matrix, dither, floyd_steinberg, ordered_dither


class TestBayerMatrix:
    def test_base_case(self) -> None:
        np.testing.assert_array_equal(bayer_matrix(1), [[0, 2], [3, 1]])

    def test_recursive_step_is_a_permutation(self) -> None:
        m = bayer_matrix(2)
        assert m.shape == (4, 4)
        assert sorted(m.flatten().tolist()) == list(range(16))

    def test_8x8_is_a_permutation(self) -> None:
        m = bayer_matrix(3)
        assert m.shape == (8, 8)
        assert sorted(m.flatten().tolist()) == list(range(64))

    def test_rejects_bad_power(self) -> None:
        with pytest.raises(ValueError, match="power"):
            bayer_matrix(0)


class TestOrderedDither:
    def test_output_is_binary(self) -> None:
        rng = np.random.default_rng(3)
        mask = ordered_dither(rng.random((32, 32), dtype=np.float32))
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 1}

    def test_mid_gray_gives_exact_half_density(self) -> None:
        signal = np.full((8, 8), 0.5, dtype=np.float32)
        assert int(ordered_dither(signal).sum()) == 32

    def test_extremes(self) -> None:
        assert ordered_dither(np.zeros((16, 16), dtype=np.float32)).sum() == 0
        assert ordered_dither(np.ones((16, 16), dtype=np.float32)).sum() == 16 * 16

    def test_density_tracks_signal_level(self) -> None:
        for level in (0.25, 0.5, 0.75):
            signal = np.full((64, 64), level, dtype=np.float32)
            density = float(ordered_dither(signal).mean())
            assert density == pytest.approx(level, abs=0.02)

    def test_non_multiple_of_tile_shape(self) -> None:
        signal = np.full((13, 21), 0.5, dtype=np.float32)
        mask = ordered_dither(signal)
        assert mask.shape == (13, 21)


class TestFloydSteinberg:
    def test_output_is_binary(self) -> None:
        rng = np.random.default_rng(11)
        mask = floyd_steinberg(rng.random((16, 16), dtype=np.float32))
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 1}

    def test_preserves_mean_of_constant_field(self) -> None:
        signal = np.full((32, 32), 0.3, dtype=np.float32)
        assert float(floyd_steinberg(signal).mean()) == pytest.approx(0.3, abs=0.03)

    def test_preserves_mean_of_gradient(self) -> None:
        gradient = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (64, 1))
        mask = floyd_steinberg(gradient)
        assert float(mask.mean()) == pytest.approx(float(gradient.mean()), abs=0.02)
        # The dark side must be sparser than the bright side.
        assert float(mask[:, :16].mean()) < float(mask[:, -16:].mean())


class TestDispatch:
    def test_dispatch_selects_algorithm(self) -> None:
        signal = np.full((8, 8), 0.5, dtype=np.float32)
        np.testing.assert_array_equal(dither(signal, DitherMode.BAYER), ordered_dither(signal))
        np.testing.assert_array_equal(
            dither(signal, DitherMode.FLOYD_STEINBERG), floyd_steinberg(signal)
        )
