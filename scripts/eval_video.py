"""Score video plans against the tape measurements in ground_truth.json (the Data/ flat).

    uv run python scripts/eval_video.py PLAN[:ROOM] [PLAN[:ROOM] ...] [--tol 0.03]
    uv run python scripts/eval_video.py --repeat PLAN_A PLAN_B

PLAN is a plan.json. The truth room comes from the room's name (B1, B2, BR, H1, H2 and their
``_2`` repeats), or from ROOM (a key of ground_truth.json) when the name is not enough. A plan with
several rooms is scored on its largest. For each tape wall the nearest-length plan wall is taken
(one plan wall per tape wall) and passes within ``--tol`` (video: 3%). Ceiling passes within 1.5 cm,
openings within 2 cm (a missed or a phantom opening is a miss); every row also says whether the
tape value lies inside the plan's own [lo, hi]. ``--repeat`` compares the walls of two captures of
one room: within 1 cm or 0.5% per wall, and the ceilings within 1 cm. Only doors are scored, not
windows. Always exits 0: it reports, it does not gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from floorfathom.schema import CapturePlan, Measurement, RoomPlan

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "b1": "bedroom_b1", "bedroom_b1": "bedroom_b1",
    "b2": "master_bedroom_b2", "master_bedroom_b2": "master_bedroom_b2",
    "br": "bathroom", "bathroom": "bathroom",
    "h1": "hall_kitchen", "h2": "hall_kitchen", "hall": "hall_kitchen", "hall_kitchen": "hall_kitchen",
}
CEILING_GATE, OPENING_GATE = 1.5, 2.0  # cm


def truth_room(key: str) -> dict:
    """Tape values of one room in cm: walls (those measured), ceiling, area in m2 or None, door widths."""
    r = json.loads((ROOT / "ground_truth.json").read_text())["rooms"][key]
    if "walls" in r:
        walls = [(w["id"], float(w["length"])) for w in r["walls"]]
    else:
        walls = [(f"{n}{i}", float(r[n])) for i, n in enumerate(["width", "length", "width", "length"])]
    area = r["width"] * r["length"] / 1e4 if "width" in r and "length" in r and "walls" not in r else None
    doors = [(o["id"], float(o["width"])) for o in r["openings"] if o["type"] == "door"]
    return {"walls": walls, "ceiling": float(r["ceiling_height"]), "area": area, "doors": doors}


def largest_room(plan: CapturePlan) -> RoomPlan:
    return max(plan.rooms, key=lambda r: r.floor_area.value or 0.0)


def assign(a: list[float], b: list[float]) -> list[tuple[int, int]]:
    """One-to-one pairing of a with b that minimises the total absolute difference."""
    if not a or not b:
        return []
    i, j = linear_sum_assignment(np.abs(np.subtract.outer(a, b)))
    return sorted(zip(i.tolist(), j.tolist()))


def inside(t: float, m: Measurement) -> str:
    return "n/a" if m.lo is None or m.hi is None else ("yes" if m.lo <= t <= m.hi else "NO")


def _split(spec: str) -> tuple[str, str]:
    """PLAN[:ROOM] to (plan path, room key); a colon that is part of a drive letter is not a separator."""
    path, sep, key = spec.rpartition(":")
    return (path, key) if sep and key and "\\" not in key and "/" not in key else (spec, "")


def _resolve(spec: str) -> tuple[Path, str]:
    path, key = _split(spec)
    plan = CapturePlan.model_validate_json(Path(path).read_text())
    name = (key or largest_room(plan).name or plan.capture).lower().removesuffix("_2")
    if name not in ALIASES and key not in ALIASES.values():
        raise SystemExit(f"{path}: cannot tell which tape room '{name}' is; give PLAN:ROOM with one of {sorted(set(ALIASES.values()))}")
    return Path(path), ALIASES.get(name, key)


def score(spec: str, tol: float) -> None:
    path, key = _resolve(spec)
    plan = CapturePlan.model_validate_json(path.read_text())
    room, t = largest_room(plan), truth_room(key)
    print(f"\n### {path} -> {key} (plan room {room.id}, {room.floor_area.value:.2f} m2 of {len(plan.rooms)} room(s); flags: {', '.join(room.flags) or 'none'})\n")
    print("| item | tape (cm) | plan (cm) | error | pass | tape inside [lo, hi] |")
    print("|---|---|---|---|---|---|")
    lengths = [w.length.value * 100 for w in room.walls]
    got = dict(assign([v for _, v in t["walls"]], lengths))
    n_ok = 0
    for i, (wid, tape) in enumerate(t["walls"]):
        if i not in got:
            print(f"| wall {wid} | {tape:.1f} | none | | MISS | |")
            continue
        w = room.walls[got[i]]
        err = lengths[got[i]] / tape - 1
        ok = abs(err) <= tol
        n_ok += ok
        print(f"| wall {wid} | {tape:.1f} | {lengths[got[i]]:.1f} | {err * 100:+.1f}% | {'ok' if ok else 'FAIL'} | {inside(tape / 100, w.length)} |")
    c = room.ceiling_height
    if c.value is None:
        print(f"| ceiling | {t['ceiling']:.1f} | not measured | | MISS | |")
    else:
        e = c.value * 100 - t["ceiling"]
        print(f"| ceiling | {t['ceiling']:.1f} | {c.value * 100:.1f} | {e:+.1f} cm | {'ok' if abs(e) <= CEILING_GATE else 'FAIL'} | {inside(t['ceiling'] / 100, c)} |")
    if t["area"] is None or room.floor_area.value is None:
        print("| area | not derivable from the tape | | | | |")
    else:
        a = room.floor_area
        print(f"| area (m2) | {t['area']:.2f} | {a.value:.2f} | {(a.value / t['area'] - 1) * 100:+.1f}% | | {inside(t['area'], a)} |")
    widths = [o.width.value * 100 for o in room.openings if o.width.value is not None]
    pairs = assign([v for _, v in t["doors"]], widths)
    matched_plan = {j for _, j in pairs}
    ok_open = 0
    for i, j in pairs:
        oid, tape = t["doors"][i]
        e = widths[j] - tape
        ok_open += abs(e) <= OPENING_GATE
        print(f"| door {oid} | {tape:.1f} | {widths[j]:.1f} | {e:+.1f} cm | {'ok' if abs(e) <= OPENING_GATE else 'FAIL'} | {inside(tape / 100, room.openings[j].width)} |")
    for i in set(range(len(t["doors"]))) - {i for i, _ in pairs}:
        print(f"| door {t['doors'][i][0]} | {t['doors'][i][1]:.1f} | none | | MISSED | |")
    for j in set(range(len(widths))) - matched_plan:
        print(f"| phantom opening | none | {widths[j]:.1f} | | PHANTOM | |")
    n_open = len(t["doors"]) + len(set(range(len(widths))) - matched_plan)
    print(f"\nwalls within {tol:.0%}: {n_ok}/{len(t['walls'])}; openings within {OPENING_GATE:.0f} cm (missed and phantom count): {ok_open}/{n_open}")


def repeat(spec_a: str, spec_b: str) -> None:
    ra, rb = (largest_room(CapturePlan.model_validate_json(Path(_split(s)[0]).read_text())) for s in (spec_a, spec_b))
    la, lb = [w.length.value * 100 for w in ra.walls], [w.length.value * 100 for w in rb.walls]
    print(f"\n### repeatability {spec_a} vs {spec_b}: {len(la)} and {len(lb)} walls\n")
    print("| wall A (cm) | wall B (cm) | difference | pass (1 cm or 0.5%) |")
    print("|---|---|---|---|")
    n_ok = 0
    for i, j in assign(la, lb):
        d = lb[j] - la[i]
        ok = abs(d) <= max(1.0, 0.005 * la[i])
        n_ok += ok
        print(f"| {la[i]:.1f} | {lb[j]:.1f} | {d:+.1f} cm | {'ok' if ok else 'FAIL'} |")
    total = max(len(la), len(lb))
    print(f"\nwalls agreeing: {n_ok}/{total} (walls with no partner count as a miss)")
    if ra.ceiling_height.value is not None and rb.ceiling_height.value is not None:
        d = (rb.ceiling_height.value - ra.ceiling_height.value) * 100
        print(f"ceiling spread: {d:+.1f} cm -> {'ok' if abs(d) <= 1.0 else 'FAIL'} (gate 1 cm)")
    else:
        print("ceiling spread: a ceiling was not measured in one of the captures")


def main(argv: list[str]) -> None:
    tol = 0.03
    if "--tol" in argv:
        i = argv.index("--tol")
        tol = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    if argv and argv[0] == "--repeat" and len(argv) == 3:
        repeat(argv[1], argv[2])
    elif argv and not argv[0].startswith("--"):
        for spec in argv:
            score(spec, tol)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
