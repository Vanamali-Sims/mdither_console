"""Webcam capture wrapper. This is an I/O edge: the only vision module using OpenCV."""

from __future__ import annotations

from types import TracebackType

import cv2
import numpy as np
import numpy.typing as npt


class Camera:
    """Context-managed wrapper over ``cv2.VideoCapture``.

    Yields BGR uint8 frames, optionally mirrored so on-screen movement matches
    the user's own left/right.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        mirror: bool = True,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the device and request the configured resolution."""
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            msg = f"could not open camera index {self._index}"
            raise RuntimeError(msg)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture

    def read(self) -> npt.NDArray[np.uint8] | None:
        """Grab one frame, or ``None`` if the device produced nothing."""
        if self._capture is None:
            msg = "camera is not open"
            raise RuntimeError(msg)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self._mirror:
            frame = cv2.flip(frame, 1)
        result: npt.NDArray[np.uint8] = frame
        return result

    def close(self) -> None:
        """Release the device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
