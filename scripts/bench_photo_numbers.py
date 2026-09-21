"""Print the photo-tier plan's geometry (walls, ceiling, area, openings) with intervals, for the
benchmark report. Reads a finished plan.json (per-room, in out/<name>/plan.json or rooms/<room>.json).
"""
import sys
import json
from pathlib import Path


def show(path: str):
    p = json.loads(Path(path).read_text())
    print(f"\n### {path}  (capture={p.get('capture')})")
    for room in p["rooms"]:
        print(f"  room {room['id']} '{room['name']}' frame={room['frame']}")
        ch = room["ceiling_height"]
        fa = room["floor_area"]
        def fmt(m):
            if m is None or m.get("value") is None:
                return "n/a"
            lo, hi = m.get("lo"), m.get("hi")
            iv = f" [{lo:.3g}, {hi:.3g}]" if lo is not None and hi is not None else ""
            return f"{m['value']:.4g}{iv} {m.get('unit','')}"
        print(f"    ceiling_height: {fmt(ch)}")
        print(f"    floor_area:     {fmt(fa)}")
        print(f"    walls ({len(room['walls'])}):")
        for w in room["walls"]:
            print(f"      {w['id']:>10}: len={fmt(w['length'])}  evidence={w.get('evidence')}")
        print(f"    openings ({len(room['openings'])}):")
        for o in room["openings"]:
            print(f"      {o['id']:>10}: {o.get('kind')} width={fmt(o['width'])}")
        flags = room.get("flags", [])
        print(f"    flags: {flags}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        show(a)
