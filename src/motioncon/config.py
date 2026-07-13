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

    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    DOUBLE_SWIPE_LEFT = auto()
    SELECT = auto()


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunable parameters, grouped in one immutable place.

    Units: normalized image coordinates are in ``[0, 1]`` with the origin at the
    top-left; velocities are in normalized units per second; times in seconds.
    """

    # Camera (I/O edge)
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror: bool = True

    # Difference / trails
    trails_decay: float = 0.88
    noise_floor: float = 0.06
    signal_gain: float = 4.0

    # Motion analysis
    motion_threshold: float = 0.12
    min_energy: float = 1e-4
    blob_cell_size: int = 16
    blob_window_cells: int = 3
    velocity_window: int = 5

    # Gestures
    swipe_velocity_threshold: float = 1.2
    swipe_release_factor: float = 0.5
    event_cooldown_s: float = 0.35
    double_swipe_window_s: float = 0.9
    cursor_smoothing: float = 0.5

    # Push-to-select
    select_energy_threshold: float = 0.18
    select_steady_speed: float = 0.6

    # Rendering
    dither_mode: DitherMode = DitherMode.BAYER
    color_scheme: ColorScheme = ColorScheme.GREEN_ON_BLACK
    cell_size: int = 8
    dot_radius: float = 0.38

    # Telemetry
    telemetry_path: str = "telemetry.jsonl"
