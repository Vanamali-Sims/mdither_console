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
        assert 0.0 < s.motion_threshold < 1.0
        assert s.cell_size >= 1
        assert 0.0 < s.dot_radius <= 0.5
        assert s.event_cooldown_s > 0.0
        assert s.double_swipe_window_s > s.event_cooldown_s

    def test_immutable(self) -> None:
        s = Settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.cell_size = 4  # type: ignore[misc]


class TestEnums:
    def test_all_gesture_events_exist(self) -> None:
        names = {e.name for e in Event}
        assert names == {
            "SWIPE_LEFT",
            "SWIPE_RIGHT",
            "SWIPE_UP",
            "SWIPE_DOWN",
            "DOUBLE_SWIPE_LEFT",
            "SELECT",
        }

    def test_color_schemes_expose_rgb_pairs(self) -> None:
        for scheme in ColorScheme:
            assert len(scheme.foreground) == 3
            assert len(scheme.background) == 3
            assert all(0 <= c <= 255 for c in scheme.foreground + scheme.background)

    def test_dither_modes(self) -> None:
        assert {m.name for m in DitherMode} == {"BAYER", "FLOYD_STEINBERG"}
