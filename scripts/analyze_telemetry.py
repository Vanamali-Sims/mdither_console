"""Quick telemetry stats for debugging tracking/cursor."""
import json
from pathlib import Path

path = Path("telemetry.jsonl")
lines = path.read_text().splitlines()
frames = []
for line in lines:
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        continue
    if o.get("kind") == "frame":
        frames.append(o)

tail = frames[-800:]
n = len(tail)
locked = sum(1 for f in tail if f.get("track_locked"))
cent = sum(1 for f in tail if f.get("centroid"))
area10 = sum(1 for f in tail if f.get("blob_area", 0) >= 0.10)
blocked = sum(
    1
    for f in tail
    if f.get("centroid") and f.get("blob_area", 0) >= 0.10 and not f.get("track_locked")
)
print(f"last {n} frames: locked={locked} centroid={cent} area>=0.1={area10}")
print(f"centroid+area but NOT locked={blocked} (cursor blocked)")
print(f"locked+centroid={sum(1 for f in tail if f.get('track_locked') and f.get('centroid'))}")
if tail:
    ts = [f["ts"] for f in tail]
    print(f"ts range {ts[0]:.1f} - {ts[-1]:.1f}, fps avg {n / (ts[-1] - ts[0]):.1f}")

all_tail = frames[-5000:]
locked5k = sum(1 for f in all_tail if f.get("track_locked"))
print(f"last 5000 frame records: locked={locked5k}/{len(all_tail)}")
