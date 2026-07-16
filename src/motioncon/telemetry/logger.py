"""Append-only JSONL event logging. This is an I/O edge module."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO


class JsonlLogger:
    """Appends timestamped records as one JSON object per line.

    Every gesture, selection, and frame metric goes through here so the later
    analysis phase has raw data to work with. The clock is injectable for
    deterministic tests. Records flush every write by default so short sessions
    and hard kills still leave telemetry on disk.
    """

    def __init__(
        self,
        path: str | Path,
        clock: Callable[[], float] = time.time,
        flush_every: int = 1,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._clock = clock
        self._flush_every = max(flush_every, 1)
        self._since_flush = 0
        self._file: TextIO | None = None

    @property
    def path(self) -> Path:
        """Absolute path of the telemetry file."""
        return self._path

    def open(self) -> None:
        """Open the target file for appending, creating parent dirs if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered so each record reaches the OS promptly.
        self._file = self._path.open("a", encoding="utf-8", buffering=1)

    def log(self, kind: str, *, flush: bool = False, **data: Any) -> None:
        """Append one record: ``{"ts": ..., "kind": ..., **data}``."""
        if self._file is None:
            msg = "logger is not open"
            raise RuntimeError(msg)
        record = {"ts": self._clock(), "kind": kind, **data}
        self._file.write(json.dumps(record) + "\n")
        self._since_flush += 1
        if flush or self._since_flush >= self._flush_every:
            self._file.flush()
            self._since_flush = 0

    def close(self) -> None:
        """Flush and close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._since_flush = 0

    def __enter__(self) -> JsonlLogger:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
