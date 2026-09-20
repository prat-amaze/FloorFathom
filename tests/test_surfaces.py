"""Surface patches from posed frames of a synthetic painted room with known damage, furniture and depth."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floorfathom.damage import detect
from floorfathom.schema import Measurement, RoomPlan, Wall
from floorfathom.surfaces import Accumulator, Frame, surface_planes

FLOOR_Y, HEIGHT = 0.0, 2.6
RGB_W, RGB_H, D_W, D_H = 320, 240, 160, 120
K = np.array([[250.0, 0, 160], [0, 250.0, 120], [0, 0, 1]])
BASE = np.array([0.86, 0.84, 0.79])


class Quad:
    def __init__(self, origin, u, v, su, sv, colour):
        self.o, self.u, self.v = np.array(origin, float), np.array(u, float), np.array(v, float)
        self.su, self.sv, self.colour = su, sv, colour
        self.n = np.cross(self.u, self.v)

    def hit(self, o, D):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.o - o) @ self.n) / (D @ self.n)
        p = o + t[:, None] * D
        s, h = (p - self.o) @ self.u, (p - self.o) @ self.v
        ok = (t > 0.05) & (s >= 0) & (s <= self.su) & (h >= 0) & (h <= self.sv)
        return np.where(ok, t, np.inf), s, h


def _plain(s, h):
    return np.tile(BASE, (len(s), 1)) * (1.0 - 0.04 * np.clip(h, 0, 2.6) / 2.6)[:, None]


def _stained(cs, ch, rs, rh):
    def f(s, h):
        c = _plain(s, h)
        a = np.clip(1.0 - (((s - cs) / rs) ** 2 + ((h - ch) / rh) ** 2), 0, 1) ** 0.5 * 0.6
        return c * (1 - a[:, None]) + np.array([0.72, 0.60, 0.40]) * a[:, None]

    return f


def _room_quads(stain=True):
    # plan (x, y) = (world x, -world z); room x 0..4, world z -3..0
    q = {
        "w0": Quad([0, 0, 0], [1, 0, 0], [0, 1, 0], 4, HEIGHT, _stained(1.8, 1.2, 0.15, 0.2) if stain else _plain),
        "w1": Quad([4, 0, 0], [0, 0, -1], [0, 1, 0], 3, HEIGHT, _plain),
        "w2": Quad([4, 0, -3], [-1, 0, 0], [0, 1, 0], 4, HEIGHT, _plain),
        "w3": Quad([0, 0, -3], [0, 0, 1], [0, 1, 0], 3, HEIGHT, _plain),
        "floor": Quad([0, FLOOR_Y, 0], [1, 0, 0], [0, 0, -1], 4, 3, lambda s, h: np.tile(BASE * 0.7, (len(s), 1))),
        "ceiling": Quad([0, HEIGHT, 0], [1, 0, 0], [0, 0, -1], 4, 3, lambda s, h: np.tile(BASE * 1.0, (len(s), 1))),
    }
    return q


def _frames(quads, shift=(0.0, 0.0, 0.0), depth=True, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    uu, vv = np.meshgrid(np.arange(RGB_W), np.arange(RGB_H))
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    for px, pz in [(1.0, -1.0), (3.0, -1.0), (1.0, -2.0), (3.0, -2.0), (2.0, -1.5)]:
        for k in range(10):
            yaw = 2 * np.pi * k / 10
            for pitch in (-15.0, 20.0):
                f = np.array([np.sin(yaw) * np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.cos(yaw) * np.cos(np.radians(pitch))])
                down = -(np.array([0, 1.0, 0]) - f[1] * f)
                down /= np.linalg.norm(down)
                R = np.stack([np.cross(down, f), down, f], axis=1)
                o = np.array([px, 1.3, pz])
                D = d_cam @ R.T
                best, colour = np.full(len(D), np.inf), np.zeros((len(D), 3))
                for q in quads.values():
                    t, s, h = q.hit(o, D)
                    closer = t < best
                    if closer.any():
                        colour[closer] = q.colour(s[closer], h[closer])
                        best = np.where(closer, t, best)
                rgb = np.clip(colour + rng.normal(0, 0.004, colour.shape), 0, 1).reshape(RGB_H, RGB_W, 3)
                z = (best.reshape(RGB_H, RGB_W))
                d = z[::2, ::2].astype(np.float32)
                d[~np.isfinite(d)] = 0
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = R, o + np.array(shift)
                frames.append(
                    Frame(
                        rgb=(rgb * 255).astype(np.uint8),
                        K=K,
                        pose=pose,
                        depth=d if depth else None,
                        undo=(lambda P, s=np.array(shift, np.float32): P + s) if any(shift) else None,
                    )
                )
    return frames


def _plan():
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    m = lambda v, u="m": Measurement(value=v, lo=v, hi=v, unit=u, method="t")
    walls = [Wall(id=f"r0_w{i}", start=a, end=b, length=m(1.0), evidence="wall_points") for i, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1]))]
    return RoomPlan(id="room_0", polygon=poly, walls=walls, ceiling_height=m(HEIGHT), floor_area=m(12.0, "m2"), openings=[])


def _detect_all(frames, monkeypatch):
    out = {}
    for ref, plane in surface_planes(_plan(), FLOOR_Y, FLOOR_Y + HEIGHT, mpp=0.01):
        acc = Accumulator(plane)
        for fr in frames:
            acc.add(fr)
        out[ref.id] = (acc.patch(), detect(acc.patch()))
    return out


@pytest.fixture(scope="module")
def stained_frames():
    return _frames(_room_quads())


def test_the_stain_lands_on_its_wall_with_its_size_and_nothing_else_is_reported(stained_frames, monkeypatch):
    res = _detect_all(stained_frames, monkeypatch)
    assert set(res) == {"r0_w0", "r0_w1", "r0_w2", "r0_w3", "room_0_ceiling", "room_0_floor"}
    patch, found = res["r0_w0"]
    assert patch.valid.mean() > 0.6
    stain = [f for f in found if f.damage_class == "water_stain"]
    assert len(stain) == 1
    rows, cols = np.nonzero(stain[0].mask)
    assert abs(cols.mean() * 0.01 - 1.8) < 0.05 and abs(rows.mean() * 0.01 - 1.2) < 0.05
    assert abs(stain[0].width[0] - 0.30) < 0.08 and abs(stain[0].height[0] - 0.40) < 0.08
    for sid in ("r0_w1", "r0_w2", "r0_w3", "room_0_ceiling", "room_0_floor"):
        assert res[sid][1] == [], (sid, [(f.damage_class, f.evidence) for f in res[sid][1]])


def test_furniture_in_front_of_a_wall_is_masked_by_depth(monkeypatch):
    q = _room_quads(stain=False)
    q["cabinet"] = Quad([2.4, 0, -0.4], [1, 0, 0], [0, 1, 0], 0.9, 1.0, lambda s, h: np.tile([0.15, 0.10, 0.08], (len(s), 1)))
    res = _detect_all(_frames(q), monkeypatch)
    patch, found = res["r0_w0"]
    assert found == []
    behind = patch.rgb[10:90, 250:330][patch.valid[10:90, 250:330]]  # the cabinet's place: h 0.1-0.9 m, s 2.5-3.3 m
    assert behind.size == 0 or behind.mean() > 0.6  # what is kept there is wall, never the dark cabinet


def test_a_pose_offset_is_undone_by_the_correction(monkeypatch):
    shift = (0.10, 0.0, 0.0)
    res = _detect_all(_frames(_room_quads(), shift=shift), monkeypatch)
    stain = [f for f in res["r0_w0"][1] if f.damage_class == "water_stain"]
    assert len(stain) == 1
    rows, cols = np.nonzero(stain[0].mask)
    assert abs(cols.mean() * 0.01 - 1.8) < 0.05


def test_without_depth_the_stain_is_still_found_and_a_clean_room_stays_clean(monkeypatch):
    res = _detect_all(_frames(_room_quads(), depth=False), monkeypatch)
    assert [f.damage_class for f in res["r0_w0"][1]] == ["water_stain"]
    clean = _detect_all(_frames(_room_quads(stain=False), depth=False), monkeypatch)
    assert all(found == [] for _p, found in clean.values())


def test_options_one_view_no_relief_and_a_finer_grid():
    frames = _frames(_room_quads())[::7]
    wall = next(p for r, p in surface_planes(_plan(), FLOOR_Y, FLOOR_Y + HEIGHT, mpp=0.02) if r.id == "r0_w0")
    default, lenient = Accumulator(wall), Accumulator(wall, min_views=1, use_relief=False)
    for fr in frames:
        default.add(fr)
        lenient.add(fr)
    a, b = default.patch(), lenient.patch()
    assert a.relief is not None and b.relief is None
    assert b.valid.sum() > a.valid.sum() > 0
    fine = next(p for r, p in surface_planes(_plan(), FLOOR_Y, FLOOR_Y + HEIGHT, mpp=0.01) if r.id == "r0_w0")
    assert fine.shape == (2 * wall.shape[0], 2 * wall.shape[1]) and Accumulator(fine).patch().m_per_px == 0.01
