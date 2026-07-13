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
    deterministic tests.
    """

    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time) -> None:
        self._path = Path(path)
        self._clock = clock
        self._file: TextIO | None = None

    def open(self) -> None:
        """Open the target file for appending, creating parent dirs if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def log(self, kind: str, **data: Any) -> None:
        """Append one record: ``{"ts": ..., "kind": ..., **data}``."""
        if self._file is None:
            msg = "logger is not open"
            raise RuntimeError(msg)
        record = {"ts": self._clock(), "kind": kind, **data}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None

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
