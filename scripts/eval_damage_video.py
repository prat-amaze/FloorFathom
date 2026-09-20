"""Measure the video damage adapter on synthetic clips with painted damage of known size.

    uv run python scripts/eval_damage_video.py [--widths 1,2,3,5] [--dists 1.5,2.5,3.5] [--noises 0.004,0.015]

For each camera distance, crack width and image noise, one clip (with a round object in front of the wall in every other
frame, seen by a fake depth model that reads 10% short) is rendered at 720 px and judged by ``assess_room`` on the
keyframes the adapter picks. Reported: the stain's width and height error, whether the crack was found (a
``structural_crack`` within 3 cm of its painted length), the frames used, and the number of regions that are neither.
This is synthetic: it says how the adapter and detector behave when poses and scale are exact, not how they do on
a real clip.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import synth_damage_video as V  # noqa: E402

from floorfathom import damage_video as D  # noqa: E402
from floorfathom.assess import assess_room  # noqa: E402

STAIN = (2.0, 1.2, 0.21, 0.25)


def run(views: D.Views, crack_len: float):
    r = V.room()
    assess_room(r, V.FLOOR_Y, V.FLOOR_Y + V.WALL_H, views.frames_for, kinds=("wall",), use_relief=False)
    stain = [d for d in r.damage if d.damage_class == "water_stain"]
    crack = [d for d in r.damage if d.damage_class == "structural_crack" and abs(d.length.value - crack_len) <= 0.03]
    extra = len(r.damage) - len(stain[:1]) - len(crack[:1])
    dw = dh = None
    if stain:
        dw, dh = (stain[0].width.value - STAIN[2]) * 100, (stain[0].height.value - STAIN[3]) * 100
    return dw, dh, bool(crack), extra, len(views._frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,2,3,5", help="crack widths in mm")
    ap.add_argument("--dists", default="1.5,2.5,3.5", help="camera distances from the wall in m")
    ap.add_argument("--noises", default="0.004,0.015", help="image noise, standard deviation on a 0..1 scale")
    args = ap.parse_args()
    print("| dist m | crack mm | noise | frames | stain dw cm | stain dh cm | crack found | other regions |")
    print("|---|---|---|---|---|---|---|---|")
    fmt = lambda x: "missed" if x is None else f"{x:+.1f}"
    for noise in [float(x) for x in args.noises.split(",")]:
        for dist in [float(x) for x in args.dists.split(",")]:
            for mm in [float(x) for x in args.widths.split(",")]:
                crack = (3.1, 1.0, 0.22, mm / 1000)
                with tempfile.TemporaryDirectory() as tmp:
                    kf, sfm, depths = V.make_clip(
                        Path(tmp), V.texture(stain=STAIN, crack=crack), poses=V.cameras(dist=dist), occluder_in=set(range(0, 14, 2)), noise=noise
                    )
                    v = D.Views(kf, sfm, D.world_poses(sfm, V.GRAVITY, V.SCALE), V.SCALE, depth=lambda rgb: depths[rgb.tobytes()])
                    dw, dh, found, extra, used = run(v, crack[2])
                print(f"| {dist} | {mm:g} | {noise} | {used} | {fmt(dw)} | {fmt(dh)} | {'yes' if found else 'no'} | {extra} |", flush=True)


if __name__ == "__main__":
    main()
