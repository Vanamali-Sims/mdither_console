"""Tests for the JSONL telemetry logger (uses tmp files, no camera)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motioncon.telemetry.logger import JsonlLogger


def fixed_clock() -> float:
    return 123.456


class TestJsonlLogger:
    def test_writes_timestamped_records(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlLogger(path, clock=fixed_clock) as log:
            log.log("gesture", event="STROKE_LEFT", cursor=[0.4, 0.6])
            log.log("frame", energy=0.02, fps=30.1)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {
            "ts": 123.456,
            "kind": "gesture",
            "event": "STROKE_LEFT",
            "cursor": [0.4, 0.6],
        }
        second = json.loads(lines[1])
        assert second["kind"] == "frame"
        assert second["energy"] == 0.02

    def test_appends_across_sessions(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlLogger(path, clock=fixed_clock) as log:
            log.log("selection", action="play")
        with JsonlLogger(path, clock=fixed_clock) as log:
            log.log("selection", action="about")

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["action"] for r in records] == ["play", "about"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "events.jsonl"
        with JsonlLogger(path, clock=fixed_clock) as log:
            log.log("frame", energy=0.0)
        assert path.exists()

    def test_logging_while_closed_raises(self, tmp_path: Path) -> None:
        logger = JsonlLogger(tmp_path / "events.jsonl")
        with pytest.raises(RuntimeError, match="not open"):
            logger.log("frame")
