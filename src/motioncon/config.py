"""Application settings and shared enums.

Everything here is plain stdlib (dataclasses + enums) so pure modules can
depend on it without pulling in any I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DitherMode(Enum):
    """Dithering algorithm used to quantize the motion signal into a dot mask."""

    BAYER = auto()
    FLOYD_STEINBERG = auto()


class ColorScheme(Enum):
    """Named foreground/background color pairs for the dot renderer (RGB)."""

    GREEN_ON_BLACK = ((40, 255, 120), (0, 0, 0))
    AMBER_ON_BLACK = ((255, 190, 40), (10, 5, 0))
    WHITE_ON_BLACK = ((235, 235, 235), (0, 0, 0))
    BRUTALIST = ((255, 255, 255), (0, 0, 0))

    @property
    def foreground(self) -> tuple[int, int, int]:
        """Dot color as an RGB triple."""
        fg: tuple[int, int, int] = self.value[0]
        return fg

    @property
    def background(self) -> tuple[int, int, int]:
        """Canvas color as an RGB triple."""
        bg: tuple[int, int, int] = self.value[1]
        return bg


class Event(Enum):
    """Discrete gesture events emitted by the control layer."""

    FLICK_UP = auto()
    FLICK_DOWN = auto()
    SWIPE_LEFT = auto()


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunable parameters, grouped in one immutable place.

    Flow magnitudes and impulses are measured in downscaled pixels per frame.
    Zone coordinates are normalized to ``[0, 1]`` from the top of the image.
    """

    # Camera (I/O edge)
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror: bool = True

    # Difference / trails
    trails_decay: float = 0.88
    noise_floor: float = 0.10
    signal_gain: float = 4.0

    # Dense optical flow
    flow_width: int = 96
    flow_height: int = 72
    flow_mag_floor: float = 0.35
    ignore_bottom: float = 0.15
    gesture_band: tuple[float, float] = (0.25, 0.85)

    # Burst→quiet directional gestures
    presence_floor: float = 0.015
    quiet_frac: float = 0.02
    settle_s: float = 0.25
    settle_mag: float = 0.35
    arm_window_s: float = 3.0
    coh_min: float = 0.60
    coherence_collapse: float = 0.35
    flick_mag: float = 0.75
    axis_dominance: float = 1.8
    impulse_thresh: float = 3.0
    stroke_max_s: float = 0.50
    refractory_s: float = 0.40
    opp_lockout_s: float = 0.70

    # Rendering
    dither_mode: DitherMode = DitherMode.BAYER
    color_scheme: ColorScheme = ColorScheme.BRUTALIST
    cell_size: int = 8
    dot_radius: float = 0.38

    # Telemetry
    telemetry_path: str = "telemetry.jsonl"
    telemetry_enabled: bool = True
    telemetry_frame_stride: int = 3
