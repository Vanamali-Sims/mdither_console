"""Tests for settings and shared enums."""

from __future__ import annotations

import dataclasses

import pytest

from motioncon.config import ColorScheme, DitherMode, Event, Settings


class TestSettings:
    def test_defaults_are_sane(self) -> None:
        s = Settings()
        assert 0.0 <= s.trails_decay <= 1.0
        assert 0.0 <= s.noise_floor < 1.0
        assert s.cell_size >= 1
        assert 0.0 < s.dot_radius <= 0.5
        assert s.flow_width > 0 and s.flow_height > 0
        assert s.flow_mag_floor > 0.0
        assert 0.0 <= s.ignore_bottom < 1.0
        assert 0.0 <= s.gesture_band[0] < s.gesture_band[1] <= 1.0
        assert s.gesture_band[1] <= 1.0 - s.ignore_bottom
        assert 0.0 < s.presence_floor < 1.0
        assert 0.0 < s.quiet_frac < 1.0
        assert s.settle_s > 0.0
        assert s.settle_mag < s.flick_mag
        assert s.arm_window_s > 0.0
        assert 0.0 <= s.coherence_collapse < s.coh_min <= 1.0
        assert s.axis_dominance >= 1.0
        assert s.impulse_thresh > 0.0
        assert s.stroke_max_s > 0.0
        assert s.refractory_s > 0.0
        assert s.opp_lockout_s >= s.refractory_s
        assert s.telemetry_frame_stride >= 1

    def test_immutable(self) -> None:
        s = Settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.cell_size = 4  # type: ignore[misc]


class TestEnums:
    def test_all_gesture_events_exist(self) -> None:
        names = {e.name for e in Event}
        assert names == {
            "FLICK_UP",
            "FLICK_DOWN",
            "SWIPE_LEFT",
        }

    def test_color_schemes_expose_rgb_pairs(self) -> None:
        for scheme in ColorScheme:
            assert len(scheme.foreground) == 3
            assert len(scheme.background) == 3
            assert all(0 <= c <= 255 for c in scheme.foreground + scheme.background)

    def test_dither_modes(self) -> None:
        assert {m.name for m in DitherMode} == {"BAYER", "FLOYD_STEINBERG"}

    def test_brutalist_scheme_exists(self) -> None:
        assert ColorScheme.BRUTALIST.foreground == (255, 255, 255)
        assert ColorScheme.BRUTALIST.background == (0, 0, 0)
