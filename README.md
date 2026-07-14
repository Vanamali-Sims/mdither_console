# motion-console

A real-time, motion-controlled interface. The webcam renders only what **moves** as a
field of dither dots (frame differencing), so a moving hand becomes the only visible
object — and the UI is navigated purely through motion: deliberate swipes step the
menu, and pushing a hand toward the camera (blob grows) selects.

## How it works

Each frame is differenced against the previous one; the difference signal feeds two
independent consumers:

- **Visualization** — the signal is accumulated into decaying trails, dithered
  (Bayer or Floyd–Steinberg) into a binary mask, and rasterized as colored dots.
- **Control** — the dominant moving blob's centroid, energy, area, and velocity become a
  `MotionState`; a debounced state machine turns that stream into discrete gesture
  events that drive a menu.

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
| Move hand (left/right zones) | Cursor locks for 2s on the spatially-scored target |
| Swipe left / right / up / down | Step the menu (must travel 33% of screen while locked) |
| Double swipe left | Back (pop one menu level) |
| Push hand toward camera | Select when the motion blob grows in size |
| `d` key | Toggle dither algorithm (Bayer / Floyd–Steinberg) |
| `q` or `Esc` | Quit |

The HUD shows motion energy, blob patch fill (area), cursor position, the last gesture event, and FPS.
Gestures, selections, and per-frame metrics are appended to `telemetry.jsonl`.

## Development

```bash
uv run pytest              # tests (no webcam needed)
uv run ruff check          # lint
uv run ruff format         # format
uv run mypy                # type check
uv run pre-commit install  # install git hooks (ruff + mypy on commit)
```

CI runs ruff, mypy, and pytest on Python 3.11 and 3.12 for every push and pull request.
