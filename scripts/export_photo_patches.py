"""Dump the unrolled wall and ceiling patches of photo rooms as test data for the damage detector.

    uv run python scripts/export_photo_patches.py Data out/damage_patches/photo [--cache-dir out/photo_new] [--reference-length-cm 31.6]

One ``<room>_<surface id>.npz`` (rgb float16 0..1, valid bool, m_per_px, kind, room, surface; the layout scripts/eval_damage_patches.py reads) and one .png preview per surface,
built exactly as ``photo_damage.assess`` builds them (one view enough, no relief). The depth cache under
``<cache-dir>/work/depth`` is reused, so after a photo run this costs no depth-model time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from floorfathom.photo_damage import KINDS, MIN_VIEWS, MPP
from floorfathom.surfaces import Accumulator, surface_planes


def _preview(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    img = rgb[..., ::-1].copy()
    img[~valid] = (img[~valid] * 0.25 + np.array([120, 0, 80]) * 0.75).astype(np.uint8)
    return cv2.flip(img, 0)  # wall height upward


def export_patches(room, out_dir: Path, mpp: float = MPP) -> list[str]:
    """Save every wall and ceiling patch of ``room`` (a ``PhotoRoom``); returns the file stems."""
    own = room.plan.rooms[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for ref, plane in surface_planes(own, room.floor_y, room.ceiling_y, mpp):
        if ref.kind not in KINDS:
            continue
        acc = Accumulator(plane, MIN_VIEWS, use_relief=False)  # one at a time: memory is tight
        for fr in room.frames:
            acc.add(fr)
        patch = acc.patch()
        rgb = np.clip(patch.rgb, 0, 1)
        stem = f"{(own.name or own.id).lower()}_{ref.id}"
        np.savez_compressed(out_dir / f"{stem}.npz", rgb=rgb.astype(np.float16), valid=patch.valid, m_per_px=patch.m_per_px,
                            kind=ref.kind, room=own.id, surface=ref.id)
        cv2.imwrite(str(out_dir / f"{stem}.png"), _preview((rgb * 255).astype(np.uint8), patch.valid))
        stems.append(stem)
    return stems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None, help="folder whose work/depth holds the depth cache (default out_dir)")
    ap.add_argument("--reference-length-cm", type=float, default=None)
    args = ap.parse_args()
    from floorfathom.photo_pipeline import photo_rooms

    ref = args.reference_length_cm / 100 if args.reference_length_cm else None
    for room in photo_rooms(args.capture, args.cache_dir or args.out_dir, reference_length_m=ref):
        if room.frames and room.floor_y is not None:
            print(room.plan.rooms[0].name, export_patches(room, args.out_dir))


if __name__ == "__main__":
    main()
