"""Tests for the pure frame-differencing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from motioncon.vision.difference import TrailsAccumulator, boost, frame_difference, to_gray


class TestToGray:
    def test_bt601_weights(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0] = (255, 0, 0)  # pure red
        frame[0, 1] = (0, 255, 0)  # pure green
        frame[1, 0] = (0, 0, 255)  # pure blue
        frame[1, 1] = (255, 255, 255)

        gray = to_gray(frame)

        assert gray.shape == (2, 2)
        assert gray.dtype == np.float32
        np.testing.assert_allclose(gray[0, 0], 0.299, atol=1e-3)
        np.testing.assert_allclose(gray[0, 1], 0.587, atol=1e-3)
        np.testing.assert_allclose(gray[1, 0], 0.114, atol=1e-3)
        np.testing.assert_allclose(gray[1, 1], 1.0, atol=1e-3)

    def test_two_dimensional_input_rescaled(self) -> None:
        frame = np.full((3, 3), 128, dtype=np.uint8)
        gray = to_gray(frame)
        np.testing.assert_allclose(gray, 128 / 255, atol=1e-6)

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            to_gray(np.zeros((2, 2, 4), dtype=np.uint8))


class TestFrameDifference:
    def test_known_difference(self) -> None:
        a = np.array([[0.2, 0.8]], dtype=np.float32)
        b = np.array([[0.5, 0.5]], dtype=np.float32)
        diff = frame_difference(a, b)
        np.testing.assert_allclose(diff, [[0.3, 0.3]], atol=1e-6)

    def test_symmetric_and_bounded(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.random((16, 16), dtype=np.float32)
        b = rng.random((16, 16), dtype=np.float32)
        d1, d2 = frame_difference(a, b), frame_difference(b, a)
        np.testing.assert_array_equal(d1, d2)
        assert float(d1.min()) >= 0.0
        assert float(d1.max()) <= 1.0

    def test_noise_floor_zeroes_small_values(self) -> None:
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.02, 0.5]], dtype=np.float32)
        diff = frame_difference(a, b, noise_floor=0.05)
        np.testing.assert_allclose(diff, [[0.0, 0.5]], atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            frame_difference(np.zeros((2, 2), dtype=np.float32), np.zeros((3, 3), dtype=np.float32))


class TestBoost:
    def test_amplifies_and_clips(self) -> None:
        signal = np.array([[0.1, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(boost(signal, 4.0), [[0.4, 1.0]], atol=1e-6)


class TestTrailsAccumulator:
    def test_first_update_copies_signal(self) -> None:
        trails = TrailsAccumulator(decay=0.9)
        signal = np.array([[0.5, 0.0]], dtype=np.float32)
        out = trails.update(signal)
        np.testing.assert_array_equal(out, signal)
        # The buffer must be an independent copy.
        signal[0, 0] = 0.0
        assert trails.value is not None
        assert trails.value[0, 0] == pytest.approx(0.5)

    def test_decays_where_motion_stops(self) -> None:
        trails = TrailsAccumulator(decay=0.5)
        trails.update(np.array([[1.0]], dtype=np.float32))
        zero = np.zeros((1, 1), dtype=np.float32)
        assert trails.update(zero)[0, 0] == pytest.approx(0.5)
        assert trails.update(zero)[0, 0] == pytest.approx(0.25)
        assert trails.update(zero)[0, 0] == pytest.approx(0.125)

    def test_new_motion_wins_over_decayed_trail(self) -> None:
        trails = TrailsAccumulator(decay=0.9)
        trails.update(np.array([[0.3]], dtype=np.float32))
        out = trails.update(np.array([[0.8]], dtype=np.float32))
        assert out[0, 0] == pytest.approx(0.8)

    def test_reset_clears_buffer(self) -> None:
        trails = TrailsAccumulator(decay=0.9)
        trails.update(np.ones((2, 2), dtype=np.float32))
        trails.reset()
        assert trails.value is None
        out = trails.update(np.full((2, 2), 0.1, dtype=np.float32))
        np.testing.assert_allclose(out, 0.1, atol=1e-6)

    def test_invalid_decay_rejected(self) -> None:
        with pytest.raises(ValueError, match="decay"):
            TrailsAccumulator(decay=1.5)
