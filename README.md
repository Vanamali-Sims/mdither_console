# motion-console

A real-time, motion-controlled interface. The webcam renders only what **moves** as a
field of dither dots, while dense optical flow detects short, coherent hand flicks.

## How it works

Each camera frame feeds two independent paths:

- **Visualization** — the signal is accumulated into decaying trails, dithered
  (Bayer or Floyd–Steinberg) into a binary mask, and rasterized as colored dots.
- **Control** — dense Farnebäck flow is summarized as global direction, coherence, and
  active area inside a horizontal gesture band. A presence-gated state machine suppresses
  the hand raise, waits for a settle, then recognizes one directional burst.

See [docs/architecture.md](docs/architecture.md) for the module map and design rules.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run motioncon            # default camera
uv run motioncon --camera 1 # pick a device index
```

## Controls

| Input | Effect |
| --- | --- |
| Raise hand into the band, settle, flick up | Previous menu item |
| Raise hand into the band, settle, flick down | Next menu item |
| Raise hand into the band, settle, swipe left | Back |
| `d` key | Toggle dither algorithm (Bayer / Floyd–Steinberg) |
| `q` or `Esc` | Quit |

The camera is expected to point above the keyboard. Only normalized rows `0.25..0.85`
feed control; the bottom 15% and the face/head region above the band are ignored.
The HUD shows the band, detector phase, flow coherence/magnitude, impulse progress,
last gesture, and FPS. Flow metrics are logged every third frame.

## Reliability harness

Record 2–5 second clips under the supported gesture and distractor labels:

```bash
uv run python tools/record_clips.py raise_settle_flick_up --duration 3
uv run python tools/record_clips.py typing --duration 5
```

Replay all clips, print precision/recall, distractor false-positive rates and latency,
and optionally store a regression baseline:

```bash
uv run python tools/eval_detector.py
uv run python tools/eval_detector.py --write-baseline
```

Clip videos and per-frame timestamp sidecars live under `data/clips/<label>/`.

## Development

```bash
uv run pytest              # tests (no webcam needed)
uv run ruff check          # lint
uv run ruff format         # format
uv run mypy                # type check
uv run pre-commit install  # install git hooks (ruff + mypy on commit)
```

CI runs ruff, mypy, and pytest on Python 3.11 and 3.12 for every push and pull request.
