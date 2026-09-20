"""Spike: does monocular metric depth work on the iPhone HEIC stills? Not part of the pipeline.

Answers four questions before code depends on them:
  (a) do the HEICs decode upright at 3024x4032, and what does EXIF say afterwards;
  (b) is the model output z-depth or ray distance (the right reading fits the floor better);
  (c) full portrait input vs two square crops (the video spike's -32% ceiling used squares);
  (d) CPU seconds per image and the depth range.
Ceiling height is estimated per image as the distance between a floor plane (bottom rows) and
a parallel ceiling plane (top rows), and compared with the tape measure.

    uv run python scripts/spike_photo_depth.py [--rooms b2 BR] [--save out/spike_depth]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pillow_heif
from PIL import Image

from floorfathom.models import DEPTH, resolve

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parents[1]
CEILING_M = {"b1": 2.79, "b2": 2.79, "BR": 2.20, "Hall": 2.79}
LONG_SIDE = 1008  # working size (long side), a quarter of the 4032 px original
DIAG_35MM = 43.267  # diagonal of a 36 x 24 mm frame


def load(path: Path):
    im = Image.open(path)
    exif = im.getexif()
    f35 = exif.get_ifd(0x8769).get(0xA405)
    info = dict(decoded=im.size, orientation=exif.get(0x0112), f35=float(f35) if f35 else None)
    s = LONG_SIDE / max(im.size)
    return im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.BICUBIC), info


def load_model():
    import torch
    from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

    torch.set_num_threads(1)
    folder = resolve(DEPTH)
    proc = DPTImageProcessorPil.from_pretrained(folder)
    return proc, AutoModelForDepthEstimation.from_pretrained(folder, local_files_only=True).eval()


def predict(model, img: Image.Image) -> np.ndarray:
    import torch

    proc, net = model
    with torch.no_grad():
        out = net(**proc(images=img, return_tensors="pt"))
    return proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]["predicted_depth"].numpy()


def predict_squares(model, img: Image.Image) -> np.ndarray:
    """Top and bottom square crops, averaged where they overlap (what the video spike did)."""
    w, h = img.size
    top, bot = predict(model, img.crop((0, 0, w, w))), predict(model, img.crop((0, h - w, w, h)))
    acc, cnt = np.zeros((h, w)), np.zeros((h, w))
    acc[:w] += top; cnt[:w] += 1
    acc[h - w:] += bot; cnt[h - w:] += 1
    return acc / cnt


def backproject(depth: np.ndarray, fx: float, ray: bool) -> np.ndarray:
    h, w = depth.shape
    u = (np.arange(w) - w / 2 + 0.5) / fx
    v = (np.arange(h) - h / 2 + 0.5) / fx
    uu, vv = np.meshgrid(u, v)
    z = depth / np.sqrt(1 + uu**2 + vv**2) if ray else depth
    return np.stack([uu * z, vv * z, z], -1)


def ransac_plane(p: np.ndarray, thresh: float, iters: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = (None, None, 0)
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        m = np.abs((p - a) @ n) < thresh
        if m.sum() > best[2]:
            best = (n, a, int(m.sum()))
    n, a, _ = best
    m = np.abs((p - a) @ n) < thresh
    return n, float(n @ p[m].mean(0)), m


def measure(depth: np.ndarray, fx: float, ray: bool) -> dict:
    pts = backproject(depth, fx, ray)
    h = depth.shape[0]
    band = int(0.2 * h)
    sub = lambda rows: pts[rows][::4, ::4].reshape(-1, 3)
    floor, ceil = sub(slice(h - band, h)), sub(slice(0, band))
    nf, of, mf = ransac_plane(floor, 0.03)
    nc, oc, mc = ransac_plane(ceil, 0.03)
    if nf @ nc < 0:  # same orientation, so the offsets are comparable
        nc, oc = -nc, -oc
    parallel = nf @ nc > np.cos(np.radians(15))
    return dict(floor_inliers=mf.mean(), ceiling=abs(oc - of) if parallel else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=list(CEILING_M))
    ap.add_argument("--save", type=Path, default=ROOT / "out" / "spike_depth")
    args = ap.parse_args()
    model = load_model()
    args.save.mkdir(parents=True, exist_ok=True)
    print(f"{'image':<14}{'decoded':<12}{'orient':<7}{'f35':<5}{'mode':<7}{'sec':>5} {'depth p5/50/95 (m)':<20}"
          f"{'floor inl z/ray':<17}{'ceiling z/ray (m)':<20}")
    per_room: dict[str, dict[str, list[float]]] = {}
    for room in args.rooms:
        for path in sorted((ROOT / "Data" / "images" / room).glob("*.HEIC")):
            img, info = load(path)
            fx = (info["f35"] or 22.0) * float(np.hypot(*img.size)) / DIAG_35MM
            for mode, fn in (("full", predict), ("square", predict_squares)):
                t = time.time()
                d = fn(model, img)
                sec = time.time() - t
                z, r = measure(d, fx, False), measure(d, fx, True)
                fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
                p = np.percentile(d, [5, 50, 95])
                print(f"{path.stem:<14}{str(info['decoded']):<12}{str(info['orientation']):<7}{info['f35']!s:<5}{mode:<7}{sec:5.1f} "
                      f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}".ljust(66) + f"{z['floor_inliers']:.2f}/{r['floor_inliers']:.2f}".ljust(17)
                      + f"{fmt(z['ceiling'])}/{fmt(r['ceiling'])}")
                for k, v in (("z", z["ceiling"]), ("ray", r["ceiling"])):
                    if v is not None:
                        per_room.setdefault(room, {}).setdefault(f"{mode}/{k}", []).append(v)
                if path == sorted(path.parent.glob("*.HEIC"))[0] and mode == "full":
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(1, 2, figsize=(8, 5.5))
                    ax[0].imshow(img); ax[1].imshow(d, cmap="turbo"); ax[0].axis("off"); ax[1].axis("off")
                    fig.savefig(args.save / f"{room}_{path.stem}.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    print("\nmedian ceiling height per room vs tape (ratio = estimate / truth)")
    for room, modes in per_room.items():
        for k, v in modes.items():
            print(f"  {room:<5}{k:<12}n={len(v):<3}median {np.median(v):.2f} m  ratio {np.median(v) / CEILING_M[room]:.2f}  spread {np.std(v) / np.median(v):.0%}")


if __name__ == "__main__":
    main()
