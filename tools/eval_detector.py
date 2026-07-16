"""Replay labeled clips through dense flow and report detector reliability."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from motioncon.config import Event, Settings
from motioncon.control.flick import FlickDetector
from motioncon.vision.difference import to_gray
from motioncon.vision.flow import FlowAnalyzer

EXPECTED = {
    "stroke_left": Event.STROKE_LEFT,
    "stroke_right": Event.STROKE_RIGHT,
}
DISTRACTORS = {"typing", "head_turn", "lean_in_out", "raise_and_lower_only"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True, slots=True)
class Detection:
    event: Event
    timestamp: float


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    clips: int
    precision: float
    recall: float
    false_positive_rate: float
    ambiguous_rate: float
    latency_ms: float | None


def make_pipeline(settings: Settings) -> tuple[FlowAnalyzer, FlickDetector]:
    """Build the same flow and detector configuration used by the live app."""
    flow = FlowAnalyzer(
        width=settings.flow_width,
        height=settings.flow_height,
        mag_floor=settings.flow_mag_floor,
        gesture_band=settings.gesture_band,
        ignore_bottom=settings.ignore_bottom,
    )
    detector = FlickDetector(
        presence_floor=settings.presence_floor,
        quiet_frac=settings.quiet_frac,
        settle_s=settings.settle_s,
        settle_mag=settings.settle_mag,
        settle_quiet_frac=settings.settle_quiet_frac,
        arm_window_s=settings.arm_window_s,
        capture_floor=settings.capture_floor,
        reentry_mag=settings.reentry_mag,
        burst_quiet_s=settings.burst_quiet_s,
        burst_max_s=settings.burst_max_s,
        throw_impulse=settings.throw_impulse,
        coh_min=settings.coh_min,
        refractory_s=settings.refractory_s,
    )
    return flow, detector


def replay_clip(path: Path, settings: Settings) -> list[Detection]:
    """Replay one clip using preserved timestamps when available."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"could not open clip: {path}"
        raise RuntimeError(msg)

    metadata_path = path.with_suffix(".timestamps.json")
    timestamps: list[dict[str, float]] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timestamps = metadata.get("frame_timestamps", [])
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0:
        fps = 30.0

    flow, detector = make_pipeline(settings)
    detections: list[Detection] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = (
                float(timestamps[index]["elapsed_s"]) if index < len(timestamps) else index / fps
            )
            state = flow.analyze(to_gray(frame), timestamp)
            detections.extend(Detection(event, timestamp) for event in detector.update(state))
            index += 1
    finally:
        capture.release()
    return detections


def discover_clips(root: Path) -> list[tuple[str, Path]]:
    """Return all recognized videos beneath label directories."""
    clips: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            clips.append((path.parent.name, path))
    return clips


def evaluate(
    clips: list[tuple[str, Path]], settings: Settings
) -> tuple[dict[str, LabelMetrics], dict[str, dict[str, int]]]:
    """Evaluate clips and return metrics plus a first-event confusion matrix."""
    grouped: dict[str, list[list[Detection]]] = {}
    confusion: dict[str, dict[str, int]] = {}
    columns = [event.name for event in Event] + ["NONE"]

    for label, path in clips:
        detections = replay_clip(path, settings)
        grouped.setdefault(label, []).append(detections)
        predicted = detections[0].event.name if detections else "NONE"
        row = confusion.setdefault(label, dict.fromkeys(columns, 0))
        row[predicted] += 1

    metrics: dict[str, LabelMetrics] = {}
    for label, results in grouped.items():
        expected = EXPECTED.get(label)
        if expected is not None:
            correct = sum(sum(item.event is expected for item in result) for result in results)
            predicted = sum(len(result) for result in results)
            detected_clips = sum(
                any(item.event is expected for item in result) for result in results
            )
            ambiguous_clips = sum(1 for result in results if not result)
            latencies = [
                next(item.timestamp for item in result if item.event is expected) * 1000.0
                for result in results
                if any(item.event is expected for item in result)
            ]
            metrics[label] = LabelMetrics(
                clips=len(results),
                precision=correct / predicted if predicted else 0.0,
                recall=detected_clips / len(results),
                false_positive_rate=0.0,
                ambiguous_rate=ambiguous_clips / len(results),
                latency_ms=statistics.mean(latencies) if latencies else None,
            )
        else:
            false_clips = sum(bool(result) for result in results)
            metrics[label] = LabelMetrics(
                clips=len(results),
                precision=1.0 if false_clips == 0 else 0.0,
                recall=1.0,
                false_positive_rate=false_clips / len(results),
                ambiguous_rate=0.0,
                latency_ms=None,
            )
    return metrics, confusion


def print_report(metrics: dict[str, LabelMetrics], confusion: dict[str, dict[str, int]]) -> None:
    """Print metrics and a compact confusion summary."""
    print("Per-label metrics")
    print(
        f"{'label':32} {'clips':>5} {'precision':>9} {'recall':>7} "
        f"{'FPR':>7} {'ambig':>7} {'latency':>10}"
    )
    for label, item in sorted(metrics.items()):
        latency = "-" if item.latency_ms is None else f"{item.latency_ms:.0f} ms"
        print(
            f"{label:32} {item.clips:5d} {item.precision:9.3f} "
            f"{item.recall:7.3f} {item.false_positive_rate:7.3f} "
            f"{item.ambiguous_rate:7.3f} {latency:>10}"
        )

    columns = [event.name for event in Event] + ["NONE"]
    print("\nConfusion summary (first event per clip)")
    print(f"{'actual':32} " + " ".join(f"{column:>12}" for column in columns))
    for label, row in sorted(confusion.items()):
        print(f"{label:32} " + " ".join(f"{row[column]:12d}" for column in columns))

    direction_labels = [label for label in EXPECTED if label in metrics]
    if direction_labels:
        print("\nPer-direction summary")
        for label in sorted(direction_labels):
            item = metrics[label]
            print(
                f"  {label}: precision={item.precision:.3f} recall={item.recall:.3f} "
                f"ambiguous={item.ambiguous_rate:.3f}"
            )

    raise_only = metrics.get("raise_and_lower_only")
    if raise_only is not None:
        print(f"\nraise_and_lower_only false-positive rate: {raise_only.false_positive_rate:.3f}")


def serializable(metrics: dict[str, LabelMetrics]) -> dict[str, Any]:
    return {"labels": {label: asdict(item) for label, item in sorted(metrics.items())}}


def find_regressions(metrics: dict[str, LabelMetrics], baseline_path: Path) -> list[str]:
    """Compare against baseline; higher latency/FPR or lower precision/recall regress."""
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("labels", {})
    current = serializable(metrics)["labels"]
    regressions: list[str] = []
    for label, old in baseline.items():
        if label not in current:
            regressions.append(f"{label}: missing from current dataset")
            continue
        new = current[label]
        for field in ("precision", "recall"):
            if float(new[field]) + 1e-9 < float(old[field]):
                regressions.append(f"{label}: {field} {new[field]:.3f} < {old[field]:.3f}")
        if float(new["false_positive_rate"]) > float(old["false_positive_rate"]) + 1e-9:
            regressions.append(
                f"{label}: false_positive_rate {new['false_positive_rate']:.3f} "
                f"> {old['false_positive_rate']:.3f}"
            )
        if "ambiguous_rate" in old and float(new.get("ambiguous_rate", 0.0)) > float(
            old["ambiguous_rate"]
        ) + 1e-9:
            regressions.append(
                f"{label}: ambiguous_rate {new['ambiguous_rate']:.3f} > {old['ambiguous_rate']:.3f}"
            )
        old_latency = old.get("latency_ms")
        new_latency = new.get("latency_ms")
        if old_latency is not None and (new_latency is None or new_latency > old_latency):
            regressions.append(f"{label}: detection latency regressed")
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("data/clips"))
    parser.add_argument("--baseline", type=Path, default=Path("data/clips/baseline.json"))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    clips = discover_clips(args.clips)
    if not clips:
        parser.error(f"no video clips found under {args.clips}")
    metrics, confusion = evaluate(clips, Settings())
    print_report(metrics, confusion)

    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(serializable(metrics), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote baseline: {args.baseline}")
        return

    regressions = find_regressions(metrics, args.baseline)
    if regressions:
        print("\nRegressions:", file=sys.stderr)
        for regression in regressions:
            print(f"- {regression}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
