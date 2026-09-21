"""Score the photo tier's damage regions against the tape-measured damage of ``ground_truth.json``.

    uv run python scripts/eval_damage_photo.py out/photo_new/plan.json [--truth ground_truth.json]

For each ground-truth damage item (class, width, height in cm) the room of the same name is searched for regions of
the matching class and the one whose width and height are closest is scored: found or not, the size error in cm and
whether the tape value lies inside the region's 95% interval. Every other region is listed, since with only two
measured items a region that is not one of them is either a false alarm or unrecorded real damage (the bathroom's
ceiling crack); the count is the false-alarm figure to read with that in mind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_OF = {"water_damage": {"water_stain"}, "structural_crack": {"structural_crack"}}
ROOM_NAMES = {"hall_kitchen": "hall", "bedroom_b1": "b1", "master_bedroom_b2": "b2", "bathroom": "br"}


def _inside(m: dict, cm: float) -> bool:
    return m["lo"] is not None and m["lo"] * 100 <= cm <= m["hi"] * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--truth", type=Path, default=Path(__file__).resolve().parents[1] / "ground_truth.json")
    args = ap.parse_args()
    plan, truth = json.loads(args.plan.read_text()), json.loads(args.truth.read_text())
    rooms = {(r.get("name") or r["id"]).lower(): r for r in plan["rooms"]}
    used: set[tuple[str, str]] = set()
    found = total = 0
    for gt_room, gt in truth["rooms"].items():
        room = rooms.get(ROOM_NAMES.get(gt_room, gt_room).lower())
        for item in gt.get("damage", []):
            total += 1
            if room is None:
                print(f"{item['id']}: room '{gt_room}' is not in the plan")
                continue
            cands = [d for d in room["damage"] if d["damage_class"] in CLASS_OF.get(item["class"], set())]
            if not cands:
                print(f"{item['id']}: NOT FOUND ({item['width']} x {item['height']} cm {item['class']}); "
                      f"regions in the room: {[(d['damage_class'], d['surface']['id']) for d in room['damage']]}")
                continue
            best = min(cands, key=lambda d: abs(d["width"]["value"] * 100 - item["width"]) + abs(d["height"]["value"] * 100 - item["height"]))
            used.add((room["id"], best["id"]))
            found += 1
            w, h = best["width"], best["height"]
            print(f"{item['id']}: found {best['damage_class']} on {best['surface']['id']} conf {best['class_confidence']:.2f}; "
                  f"width {w['value'] * 100:.1f} cm [{w['lo'] * 100:.1f}, {w['hi'] * 100:.1f}] vs tape {item['width']} "
                  f"({'inside' if _inside(w, item['width']) else 'OUTSIDE'}); "
                  f"height {h['value'] * 100:.1f} cm [{h['lo'] * 100:.1f}, {h['hi'] * 100:.1f}] vs tape {item['height']} "
                  f"({'inside' if _inside(h, item['height']) else 'OUTSIDE'})")
    print(f"\nmeasured items found: {found} of {total}")
    other = [(r.get("name") or r["id"], d) for r in plan["rooms"] for d in r["damage"] if (r["id"], d["id"]) not in used]
    print(f"other regions: {len(other)}")
    for name, d in other:
        print(f"  {name} {d['id']}: {d['damage_class']} on {d['surface']['id']} conf {d['class_confidence']:.2f}, "
              f"{d['width']['value'] * 100:.0f} x {d['height']['value'] * 100:.0f} cm, {d['evidence']}")


if __name__ == "__main__":
    main()
