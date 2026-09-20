"""Damage regions on one rectified surface: patches that look or measure unlike the rest of it.

Input is a ``Patch``: the surface unrolled flat (a wall, the ceiling or the floor) as a colour image at a known
metres per pixel, a mask of the pixels that were seen face on and unoccluded, and optionally the surface's
relief (metres off the fitted plane, from depth). Nothing here knows which tier made the patch.

Proposals are class-agnostic. A smooth colour model of the surface itself (a robust quadratic per Lab channel)
is subtracted, and what is left far above the surface's own noise is a region; thin dark ridges are found on
their own (cracks are too thin for the blob route); relief pits and broad bulges come from depth. Each region
then gets the first class whose rule it meets, else ``other_anomaly``, so a class not in the list is still
localised and measured. The rules and their numbers (``RULES`` below) are our own assumptions, tuned on
synthetic patches only: no real damage was available for the tier this was built on.

Extents carry an interval from re-cutting the region at a stricter and a looser threshold, widened by one pixel
and by the scale error of the patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

# thresholds of the rules, in Lab units (L 0-100, a and b about -100..100) and metres
RULES = {
    "z_threshold": 4.0,  # deviation from the surface, in units of its own noise
    "refine_frac": 0.5,  # a region ends where it has faded to this fraction of its own peak
    "refine_alt": (0.35, 0.65),  # the cuts that give the extent its interval
    "refine_grow": 0.03,  # m a region may grow when cut at its own fraction
    "lighting_edge_cm": 8.0,  # a colourless change with edges softer than this is lighting
    "min_area": 0.0015,  # m2: smaller colour blobs are ignored
    "core_margin": 0.10,  # m: skirting, cornices and edge misregistration are not judged
    "sigma_smooth": (0.008, 0.03),  # m, blur of the blob route: fine, and coarse for speckled clusters
    "noise_floor": (1.2, 0.8, 0.8),  # smallest noise assumed per Lab channel
    "crack_min_length": 0.08,
    "crack_min_contrast": 8.0,  # lightness below the surface: fainter ridges are texture, not cracks
    "max_area_stain": 1.0,  # m2: a stain, soot or mould can be this large ...
    "max_area_other": 0.25,  # ... anything else this large is an object or a panel, not damage
    "max_share": 0.35,  # of the surface's judged area: larger than this is a different surface, not damage
    "rect_fill": 0.88,  # a region this rectangular and axis-aligned is a door, panel or screen (an ellipse fills 0.79)
    "rect_axis_deg": 3.0,
    "rect_min_side": 0.08,
    "crack_max_breadth": 0.02,
    "crack_ridge_z": 5.0,
    "joint_straight": 0.006,  # m rms off a straight line ...
    "joint_axis_deg": 2.0,  # ... and this close to horizontal or vertical: a joint or trim, not a crack
    "relief_min": 0.025,  # m off the plane
    "relief_hole_area": 0.05,  # m2: smaller pits are holes or impacts, larger ones sag
    "relief_min_area": 0.001,
    "scale_rel_sigma": 0.02,  # relative error of the patch scale
}


@dataclass
class Patch:
    rgb: np.ndarray  # (H, W, 3) float, 0..1
    valid: np.ndarray  # (H, W) bool, seen face on and unoccluded
    m_per_px: float
    relief: np.ndarray | None = None  # (H, W) metres toward the room, nan where unknown
    kind: str = "wall"


@dataclass
class Found:
    mask: np.ndarray
    damage_class: str
    confidence: float
    evidence: str
    width: tuple[float, float, float]  # value, lo, hi in metres
    height: tuple[float, float, float]
    length: tuple[float, float, float]
    area: tuple[float, float, float]  # m2
    outline: list[tuple[float, float]]  # (col, row) pixels


def _poly_features(rows: np.ndarray, cols: np.ndarray, hh: float, ww: float) -> np.ndarray:
    y = rows / max(hh, 1.0) * 2 - 1
    x = cols / max(ww, 1.0) * 2 - 1
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def _background(lab: np.ndarray, valid: np.ndarray, mpp: float) -> np.ndarray | None:
    """Robust quadratic per Lab channel over the valid pixels: the surface as if undamaged."""
    h, w = valid.shape
    step = max(1, round(0.02 / mpp))
    hh, ww = h // step, w // step
    if hh < 4 or ww < 4:
        return None
    v = valid[: hh * step, : ww * step].astype(np.float32)
    cnt = v.reshape(hh, step, ww, step).sum((1, 3))
    ok = cnt >= 0.5 * step * step
    if ok.sum() < 200:
        return None
    cs = (lab[: hh * step, : ww * step] * v[..., None]).reshape(hh, step, ww, step, 3).sum((1, 3))
    cm = cs / np.maximum(cnt, 1)[..., None]
    rr, cc = np.mgrid[0:hh, 0:ww]
    a_all = _poly_features(rr, cc, hh - 1, ww - 1)[ok]
    bg = np.empty_like(lab)
    frr, fcc = np.mgrid[0:h, 0:w]
    a_full = _poly_features(frr / step, fcc / step, hh - 1, ww - 1)
    for c in range(3):
        y = cm[..., c][ok]
        keep = np.ones(len(y), bool)
        for _ in range(5):
            coef, *_r = np.linalg.lstsq(a_all[keep], y[keep], rcond=None)
            res = y - a_all @ coef
            s = 1.4826 * np.median(np.abs(res)) + 1e-6
            keep = np.abs(res) < 2.5 * s
            if keep.sum() < 100:
                break
        bg[..., c] = a_full @ coef
    return bg


def _smooth(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores invalid pixels (normalised convolution)."""
    v = valid.astype(np.float32)
    num = ndi.gaussian_filter(x * (v if x.ndim == 2 else v[..., None]), (sigma, sigma) + (0,) * (x.ndim - 2))
    den = ndi.gaussian_filter(v, sigma)
    den = np.maximum(den, 1e-3)
    return num / (den if x.ndim == 2 else den[..., None])


def _extent(mask: np.ndarray, mpp: float) -> tuple[float, float, float, float, list[tuple[float, float]]]:
    rows, cols = np.nonzero(mask)
    width = (cols.max() - cols.min() + 1) * mpp
    height = (rows.max() - rows.min() + 1) * mpp
    pts = np.stack([cols, rows], axis=1).astype(np.float32)
    (_, (rw, rh), _) = cv2.minAreaRect(pts)
    length = max(rw, rh, 1.0) * mpp
    if rw * rh == 0:  # a one-pixel-wide line: the box is degenerate, use the pixel span
        length = max(width, height)
    cnts, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(big, max(1.0, 0.004 / mpp), True)[:, 0, :]
    return width, height, length, mask.sum() * mpp * mpp, [(float(x), float(y)) for x, y in approx]


def _interval(base: float, alts: list[float], mpp: float, scale_rel: float) -> tuple[float, float, float]:
    lo, hi = min([base] + alts), max([base] + alts)
    rel = scale_rel * 1.96
    return base, max(0.0, lo * (1 - rel) - mpp), hi * (1 + rel) + mpp


def _region_at(labels: np.ndarray, n: int, mask: np.ndarray) -> np.ndarray | None:
    """The component of ``labels`` that overlaps ``mask`` most."""
    hit = labels[mask]
    hit = hit[hit > 0]
    if hit.size == 0:
        return None
    k = np.bincount(hit, minlength=n + 1).argmax()
    return labels == k


def _clean(mask: np.ndarray, mpp: float) -> np.ndarray:
    r = max(1, round(0.004 / mpp))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return ndi.binary_fill_holes(m.astype(bool))


def _classify(f: dict) -> tuple[str, float, str]:
    """First matching rule wins. ``f`` holds the region's measured features."""
    dl, da, db = f["dL"], f["da"], f["db"]
    area, sol, soft, busy = f["area"], f["solidity"], f["edge_cm"], f["busy"]
    ev = (
        f"dL {dl:+.0f}, da {da:+.0f}, db {db:+.0f}, area {area * 1e4:.0f} cm2, solidity {sol:.2f}, "
        f"edge width {soft:.1f} cm, texture {busy:.1f}"
    )

    def score(*margins: float) -> float:
        return float(np.clip(np.mean(np.clip(margins, 0.0, 1.0)), 0.05, 0.95))

    colourless = np.hypot(da, db) < 5  # darker or lighter without a hue shift
    whitish = np.hypot(f["a_abs"], f["b_abs"]) < 12
    if dl <= -15 and colourless and area >= 0.01 and soft >= 1.0:
        return "soot_or_fire", score((-dl - 15) / 15 + 0.5, soft / 3, area / 0.05), ev
    if dl <= -10 and busy >= 1.8 and area >= 0.002 and db < 6 and np.hypot(da, db) < 25:
        return "mould", score((-dl - 10) / 15 + 0.5, (busy - 1.8) / 2 + 0.5, 1.0 - max(db, 0) / 6), ev
    if db >= 4 and -35 <= dl <= -2 and soft >= 0.8 and area >= 0.002:
        return "water_stain", score((db - 4) / 8 + 0.5, soft / 2, (1 - abs(dl + 15) / 25)), ev
    if dl >= 8 and busy >= 1.5 and sol < 0.85:
        return "peeling_paint", score((dl - 8) / 15 + 0.5, (0.85 - sol) / 0.3 + 0.5, (busy - 1.5) / 2 + 0.5), ev
    if dl >= 5 and whitish and soft >= 0.8:
        return "efflorescence", score((dl - 8) / 15 + 0.5, soft / 2, 0.6), ev
    if dl <= -30 and soft < 0.8 and area <= 0.01:
        return "hole_or_impact", score((-dl - 30) / 20 + 0.5, 1 - soft / 0.8, 0.6), ev
    return "other_anomaly", 0.3, ev


def _is_lighting(f: dict) -> bool:
    """A soft, colourless change of lightness is a shadow or a light patch on the surface, not damage."""
    return f["edge_cm"] >= RULES["lighting_edge_cm"] and abs(f["dL"]) < 12 and max(abs(f["da"]), abs(f["db"])) < 2.5


def _is_object(mk: np.ndarray, f: dict, cls: str, mpp: float, core: np.ndarray) -> bool:
    """Doors, wardrobes, screens, lamps and panels sit on or against a wall and are not damage."""
    cap = RULES["max_area_stain"] if cls in ("water_stain", "soot_or_fire", "mould") else RULES["max_area_other"]
    if f["area"] > cap or mk.sum() > RULES["max_share"] * core.sum():
        return True
    pts = np.stack(np.nonzero(mk)[::-1], axis=1).astype(np.float32)
    (_c, (rw, rh), ang) = cv2.minAreaRect(pts)
    if min(rw, rh) * mpp < RULES["rect_min_side"] or rw * rh == 0:
        return False
    axis = min(ang % 90, 90 - ang % 90)
    return bool(mk.sum() / (rw * rh) >= RULES["rect_fill"] and axis <= RULES["rect_axis_deg"])


def _blob_features(mask, lab, bg, lsm, mpp) -> dict:
    inside = mask
    res = lab - bg
    d = res[inside].mean(axis=0)
    ab = lab[inside].mean(axis=0)
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~mask
    gy, gx = np.gradient(lsm)
    grad = np.hypot(gx, gy)
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    g = float(grad[edge].mean()) if edge.any() else 0.0
    edge_cm = abs(d[0]) / max(g, 1e-6) * mpp * 100 if abs(d[0]) > 0.5 else 0.0
    loc = lab[..., 0] - ndi.gaussian_filter(lab[..., 0], 0.01 / mpp)
    noise = max(np.std(loc[~mask & ~ring]) if (~mask & ~ring).any() else 1.0, 0.3)
    core = cv2.erode(mask.astype(np.uint8), np.ones((2 * 5 + 1,) * 2, np.uint8)).astype(bool)  # away from the outline's own edge
    busy = float(np.std(loc[core if core.sum() >= 30 else inside]) / noise)
    cnts, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(big))
    return dict(
        dL=float(d[0]), da=float(d[1]), db=float(d[2]), a_abs=float(ab[1]), b_abs=float(ab[2]),
        area=float(mask.sum() * mpp * mpp), solidity=float(mask.sum() / max(hull, 1.0)), edge_cm=float(edge_cm), busy=busy,
    )


def _line_stats(m: np.ndarray, mpp: float) -> tuple[float, float, float]:
    """Length along the principal axis, rms distance of the centre line off that axis (m) and its angle off horizontal/vertical (deg)."""
    rows, cols = np.nonzero(m)
    pts = np.stack([cols, rows], axis=1).astype(np.float64) * mpp
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    along = (pts - c) @ vt[0]
    off = (pts - c) @ vt[1]
    bins = np.floor((along - along.min()) / 0.01).astype(int)  # the centre line: mean offset per centimetre
    off = (np.bincount(bins, weights=off) / np.maximum(np.bincount(bins), 1))[np.bincount(bins) > 0]
    ang = np.degrees(np.arctan2(vt[0][1], vt[0][0])) % 180
    return float(along.max() - along.min()), float(np.sqrt(np.mean(off**2))), float(min(ang, abs(ang - 90), 180 - ang))


def _is_joint(span: float, rms: float, axis_dev: float) -> bool:
    """A long, dead-straight, axis-aligned line is a panel joint or trim; a crack wanders."""
    return span > 0.3 and rms < RULES["joint_straight"] and axis_dev < RULES["joint_axis_deg"]


def _cracks(lab_l: np.ndarray, core: np.ndarray, mpp: float) -> list[np.ndarray]:
    """Thin dark ridges: Hessian of the lightness, kept only if long, thin and not a straight axis-aligned joint."""
    ls = ndi.gaussian_filter(lab_l, 1.0)
    best = np.zeros_like(ls)
    for s in (1.0, 1.6):
        g = lambda o: ndi.gaussian_filter(lab_l, s, order=o) * s * s
        lyy, lxx, lxy = g((2, 0)), g((0, 2)), g((1, 1))
        tr, det = lxx + lyy, np.sqrt(np.maximum(((lxx - lyy) / 2) ** 2 + lxy**2, 0))
        best = np.maximum(best, tr / 2 + det)  # largest eigenvalue: positive across a dark line
    base = best[core]
    if base.size == 0:
        return []
    noise = 1.4826 * np.median(np.abs(base - np.median(base))) + 1e-6
    ridge = (best > RULES["crack_ridge_z"] * noise) & core & (ls < np.median(lab_l[core]))
    ridge = ndi.binary_closing(ridge, np.ones((3, 3)))
    lab_img, n = ndi.label(ridge, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab_img == k
        if m.sum() < 8:
            continue
        span, rms, axis_dev = _line_stats(m, mpp)
        if span < RULES["crack_min_length"] or m.sum() * mpp * mpp / span > RULES["crack_max_breadth"] * 0.6:
            continue
        if _is_joint(span, rms, axis_dev):
            continue
        out.append(m)
    return out


def _refine(seed: np.ndarray, z: np.ndarray, frac: float, mpp: float, core: np.ndarray) -> np.ndarray | None:
    """Cut a region at a fraction of its own peak: a soft stain ends where it has faded to half, not where noise begins."""
    zref = np.percentile(z[seed], 90)
    grow = max(1, round(RULES["refine_grow"] / mpp))
    reach = cv2.dilate(seed.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)).astype(bool)
    m = _clean(reach & core & (z >= max(frac * zref, 0.6 * RULES["z_threshold"])), mpp)
    lab_, n = ndi.label(m)
    return _region_at(lab_, n, seed)


def _zmaps(lab: np.ndarray, bg: np.ndarray, valid: np.ndarray, core: np.ndarray, mpp: float) -> list[np.ndarray]:
    """Deviation from the surface in units of its own noise, at a fine and a coarse blur."""
    out = []
    for sigma_m in RULES["sigma_smooth"]:
        res = _smooth(lab - bg, valid, sigma_m / mpp)
        noise = np.array(
            [max(1.4826 * np.median(np.abs(res[..., c][core] - np.median(res[..., c][core]))), RULES["noise_floor"][c]) for c in range(3)]
        )
        out.append(np.sqrt(((res / noise) ** 2).sum(axis=-1)) / np.sqrt(3.0) * 1.7)
    return out


def _relief_regions(patch: Patch, valid: np.ndarray, core: np.ndarray, mpp: float) -> list[tuple[np.ndarray, str, str]]:
    rel = np.where(np.isfinite(patch.relief), patch.relief, 0.0).astype(np.float32)
    rv = np.isfinite(patch.relief) & valid
    rs = _smooth(rel, rv, 0.02 / mpp)
    out = []
    for sign in (-1, 1):
        lab_r, nr = ndi.label(_clean((sign * rs > RULES["relief_min"]) & core & rv, mpp))
        for k in range(1, nr + 1):
            mk = lab_r == k
            a = mk.sum() * mpp * mpp
            if a < RULES["relief_min_area"] or (sign > 0 and a < RULES["relief_hole_area"]):
                continue  # a small bump is an object on the surface, not damage
            cls = "hole_or_impact" if (sign < 0 and a <= RULES["relief_hole_area"]) else "sagging_or_bulging"
            depth = float(np.abs(rs[mk]).max())
            out.append((mk, cls, f"relief {sign * depth * 100:+.1f} cm off the plane, area {a * 1e4:.0f} cm2"))
    return out


def detect(patch: Patch, scale_rel_sigma: float | None = None, report_unclassified: bool = True) -> list[Found]:
    """Damage regions of one patch. ``scale_rel_sigma`` is the patch scale's relative error (default: RULES).

    ``report_unclassified=False`` drops regions that fit no class: use it where nothing but colour says what is an
    object (no depth to mask furniture), so that only regions that look like a known kind of damage are reported.
    """
    rel = RULES["scale_rel_sigma"] if scale_rel_sigma is None else scale_rel_sigma
    mpp = patch.m_per_px
    lab = cv2.cvtColor(np.clip(patch.rgb.astype(np.float32), 0, 1), cv2.COLOR_RGB2Lab)
    valid = patch.valid.astype(bool)
    r = max(1, round(RULES["core_margin"] / mpp))
    core = cv2.erode(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    found: list[Found] = []
    bg = _background(lab, valid, mpp)
    if bg is None or core.sum() < 200:
        return found
    lsm = _smooth(lab[..., 0], valid, RULES["sigma_smooth"][0] / mpp)
    claimed: list[np.ndarray] = []  # regions already reported or deliberately dismissed

    def taken(mk: np.ndarray, share: float) -> bool:
        return any((mk & c).sum() >= share * mk.sum() for c in claimed)

    if patch.relief is not None:
        for mk, cls, ev in _relief_regions(patch, valid, core, mpp):
            found.append(_make(mk, cls, 0.6, ev, [], mpp, rel))
            claimed.append(mk)
    thr = RULES["z_threshold"]
    for zs in _zmaps(lab, bg, valid, core, mpp):
        labels, n = ndi.label(_clean((zs > thr) & core, mpp))
        for k in range(1, n + 1):
            seed = labels == k
            if seed.sum() * mpp * mpp < RULES["min_area"] or taken(seed, 0.3):
                continue
            mk = _refine(seed, zs, RULES["refine_frac"], mpp, core)
            mk = seed if mk is None else mk
            claimed.append(mk)
            alts = [m for m in (_refine(seed, zs, fr, mpp, core) for fr in RULES["refine_alt"]) if m is not None]
            f = _blob_features(mk, lab, bg, lsm, mpp)
            if _is_lighting(f):
                continue
            _w, _h, ln, ar, _o = _extent(mk, mpp)
            breadth = ar / max(ln, mpp)
            if ln >= RULES["crack_min_length"] and breadth <= 0.05 and ln >= 5 * breadth and f["db"] < 4:  # a line, not a blob
                if _is_joint(*_line_stats(mk, mpp)):
                    continue
                if f["dL"] > -RULES["crack_min_contrast"]:
                    continue  # too faint to be a crack; a light line is not damage we classify
                ev = f"thin dark line, {ln * 100:.0f} cm long, {breadth * 100:.1f} cm wide, dL {f['dL']:+.0f}"
                found.append(_make(mk, "structural_crack", 0.6, ev, alts, mpp, rel))
                continue
            cls, conf, ev = _classify(f)
            if _is_object(mk, f, cls, mpp, core) or (cls == "other_anomaly" and not report_unclassified):
                continue
            ring = cv2.dilate(mk.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool) & ~mk
            if ring.any() and (~valid[ring]).mean() > 0.3:
                conf *= 0.7
                ev += "; touches an unobserved area, extent may be cut"
            found.append(_make(mk, cls, conf, ev, alts, mpp, rel))
    for mk in _cracks(lab[..., 0], core, mpp):
        if taken(mk, 0.5):
            continue
        contrast = float(np.median(lab[..., 0][core]) - lab[..., 0][mk].mean())
        if contrast < RULES["crack_min_contrast"]:
            continue
        found.append(_make(mk, "structural_crack", 0.6, f"thin dark ridge, lightness {contrast:.0f} below the surface", [], mpp, rel))
    found.sort(key=lambda f: (np.nonzero(f.mask)[0].min(), np.nonzero(f.mask)[1].min()))
    return found


def _make(mask: np.ndarray, cls: str, conf: float, ev: str, alt_masks: list[np.ndarray], mpp: float, rel: float) -> Found:
    w, h, ln, ar, outline = _extent(mask, mpp)
    alts = [_extent(m, mpp)[:4] for m in alt_masks]
    return Found(
        mask=mask,
        damage_class=cls,
        confidence=float(conf),
        evidence=ev,
        width=_interval(w, [a[0] for a in alts], mpp, rel),
        height=_interval(h, [a[1] for a in alts], mpp, rel),
        length=_interval(ln, [a[2] for a in alts], mpp, rel),
        area=_interval(ar, [a[3] for a in alts], mpp * ln, rel),
        outline=outline,
    )
