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
| Move hand (left/right zones) | Purple cursor tracks your hand; spatial lock keeps it on target |
| Swipe left / right / up / down | Step the menu once you travel **33% of the screen** along that axis (speed does not matter) |
| Double swipe left | Back (pop one menu level) |
| Push hand toward camera (hold still) | Select when the motion blob grows in size |
| `d` key | Toggle dither algorithm (Bayer / Floyd–Steinberg) |
| `q` or `Esc` | Quit |

The HUD shows the selected menu item, blob area, lock timer, swipe travel progress (`TRAVEL 0.21/0.33`), last gesture, and FPS.
Gestures are always logged to `telemetry.jsonl`; frame metrics are logged every 3rd frame to keep FPS up.

## Development

```bash
uv run pytest              # tests (no webcam needed)
uv run ruff check          # lint
uv run ruff format         # format
uv run mypy                # type check
uv run pre-commit install  # install git hooks (ruff + mypy on commit)
```

CI runs ruff, mypy, and pytest on Python 3.11 and 3.12 for every push and pull request.
