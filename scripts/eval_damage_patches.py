"""Acceptance test of ``damage.detect`` on real rectified patches from the video tier.

    uv run python scripts/eval_damage_patches.py [--dir out/h1_patches] [--no-unclassified]

``--dir`` holds .npz patches (``rgb`` 0..1, ``valid``, ``m_per_px``) and ``labels.json``. The patches are made by
``scripts/damage_video_wall.py`` (the two labelled H1 walls) and ``scripts/damage_video_clip.py --save-patches`` (every
wall of the B1 and H1 plans). A labelled patch lists boxes ``(col0, col1, row0, row1)`` in patch pixels, row 0 at the
floor, read by eye so about 10 px accurate:

* ``must_fire``: a region of the named class must lie on the box (its centre inside the box grown by 10 px). The
  region's width and height are printed against the truth in ``what``;
* ``should_fire_if_possible``: reported as found or missed, never a failure;
* ``must_not_fire``: no region may have its centre inside the box.

A patch without labels is a negative: the B1 bedroom has no damage, and the H1 plan walls show the living room, where
none is known. Any region on it is a failure. Exit status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from floorfathom.damage import Patch, detect

GROW = 10  # px of slack around a hand-read box


def load(path: Path) -> Patch:
    z = np.load(path)
    return Patch(rgb=z["rgb"].astype(np.float32), valid=z["valid"], m_per_px=float(z["m_per_px"]))


def centre(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())


def inside(c: tuple[float, float], box: list[int], grow: int = GROW) -> bool:
    c0, c1, r0, r1 = box
    return c0 - grow <= c[0] <= c1 + grow and r0 - grow <= c[1] <= r1 + grow


def size_cm(f, mpp: float) -> str:
    return f"{f.width[0] * 100:.0f} x {f.height[0] * 100:.0f} cm"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parents[1] / "out" / "h1_patches")
    ap.add_argument("--no-unclassified", action="store_true", help="call detect with report_unclassified=False")
    args = ap.parse_args()
    labels = json.loads((args.dir / "labels.json").read_text())
    failed = 0
    print("| patch | check | result | detail |")
    print("|---|---|---|---|")
    for path in sorted(args.dir.glob("*.npz")):
        patch = load(path)
        found = detect(patch, report_unclassified=not args.no_unclassified)
        cs = [centre(f.mask) for f in found]
        lab = labels.get(path.name)
        if lab is None:
            ok = not found
            failed += not ok
            detail = "no region" if ok else "; ".join(f"{f.damage_class} {size_cm(f, patch.m_per_px)} at col {c[0]:.0f} row {c[1]:.0f}" for f, c in zip(found, cs))
            print(f"| {path.stem} | negative (no damage known) | {'PASS' if ok else 'FAIL'} | {detail} |")
            continue
        for item in lab.get("must_fire", []):
            hit = [f for f, c in zip(found, cs) if inside(c, item["box"]) and f.damage_class == item["class"]]
            wrong = [f for f, c in zip(found, cs) if inside(c, item["box"]) and f.damage_class != item["class"]]
            ok = bool(hit)
            failed += not ok
            detail = f"{hit[0].damage_class} {size_cm(hit[0], patch.m_per_px)}, truth: {item['what']}" if ok else (
                f"found as {wrong[0].damage_class} instead" if wrong else f"nothing on the box; {item['what']}")
            print(f"| {path.stem} | must fire ({item['class']}) | {'PASS' if ok else 'FAIL'} | {detail} |")
        for item in lab.get("should_fire_if_possible", []):
            hit = [f for f, c in zip(found, cs) if inside(c, item["box"])]
            print(f"| {path.stem} | should fire ({item['class']}) | {'found' if hit else 'missed'} | {item['what']} |")
        for item in lab.get("must_not_fire", []):
            bad = [(f, c) for f, c in zip(found, cs) if inside(c, item["box"], 0)]
            ok = not bad
            failed += not ok
            detail = item["what"] if ok else "; ".join(f"{f.damage_class} {size_cm(f, patch.m_per_px)} at col {c[0]:.0f} row {c[1]:.0f}" for f, c in bad)
            print(f"| {path.stem} | must not fire | {'PASS' if ok else 'FAIL'} | {detail} |")
        listed = lab.get("must_fire", []) + lab.get("should_fire_if_possible", []) + lab.get("must_not_fire", [])
        other = [(f, c) for f, c in zip(found, cs) if not any(inside(c, i["box"]) for i in listed)]
        if other:
            print(f"| {path.stem} | other regions (outside every box) | info | " + "; ".join(f"{f.damage_class} at col {c[0]:.0f} row {c[1]:.0f}" for f, c in other) + " |")
    print(f"\n{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
