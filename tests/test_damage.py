"""Damage detection on synthetic surfaces: painted damage of known place and size on a wall with gradient,
shadow, paint texture and an unobserved (occluded) area. Nothing here comes from a real capture."""

import cv2
import numpy as np

from floorfathom.damage import Patch, detect

MPP = 0.005
H, W = 480, 640  # 2.4 m x 3.2 m


def _wall(seed=0, shadow=True, occluder=True):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.array([0.86, 0.84, 0.79], np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    img *= (1.0 - 0.06 * yy / H)[..., None]  # light falls off downwards
    img *= (1.0 + 0.03 * np.sin(xx / W * 3.0))[..., None]
    if shadow:
        sh = np.exp(-(((xx - 480) / 90.0) ** 2 + ((yy - 120) / 70.0) ** 2))
        img *= (1.0 - 0.10 * sh)[..., None]  # a soft shadow, not damage
    tex = cv2.GaussianBlur(rng.normal(0, 0.012, (H, W)).astype(np.float32), (0, 0), 1.2)
    img += tex[..., None]
    valid = np.ones((H, W), bool)
    if occluder:  # furniture in front of the wall: not seen, and dark in the picture
        valid[250:480, 40:160] = False
        img[250:480, 40:160] = 0.15
    return img, valid, rng


def _blend(img, mask, colour, alpha=1.0, soft=0.0):
    a = mask.astype(np.float32)
    if soft:
        a = cv2.GaussianBlur(a, (0, 0), soft / MPP)
    a = np.clip(a * alpha, 0, 1)[..., None]
    img[:] = img * (1 - a) + np.array(colour, np.float32) * a


def _ellipse(cx, cy, rx, ry):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx / MPP), int(cy / MPP)), (int(rx / MPP), int(ry / MPP)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def _centre(f):
    rows, cols = np.nonzero(f.mask)
    return cols.mean() * MPP, rows.mean() * MPP


def _one(found, cx, cy, tol=0.06):
    near = [f for f in found if np.hypot(_centre(f)[0] - cx, _centre(f)[1] - cy) < tol]
    assert near, f"nothing near ({cx}, {cy}); found {[(f.damage_class, _centre(f)) for f in found]}"
    return max(near, key=lambda f: f.mask.sum())


def test_clean_wall_has_no_damage():
    for seed in range(3):
        img, valid, _ = _wall(seed)
        found = detect(Patch(img, valid, MPP))
        assert found == [], [(f.damage_class, f.evidence) for f in found]


def test_water_stain_class_place_and_extent():
    img, valid, _ = _wall()
    _blend(img, _ellipse(1.6, 1.0, 0.14, 0.17), (0.72, 0.60, 0.40), alpha=0.55, soft=0.02)
    f = _one(detect(Patch(img, valid, MPP)), 1.6, 1.0)
    assert f.damage_class == "water_stain"
    assert f.width[1] <= 0.28 <= f.width[2] or abs(f.width[0] - 0.28) < 0.05
    assert abs(f.height[0] - 0.34) < 0.06


def test_crack_is_found_and_long():
    img, valid, rng = _wall()
    pts = [(1.0, 0.6)]
    for _ in range(40):
        x, y = pts[-1]
        pts.append((x + 0.01 + rng.normal(0, 0.003), y + 0.008 + rng.normal(0, 0.004)))
    px = np.array([(x / MPP, y / MPP) for x, y in pts], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [px], False, 1, 2)
    _blend(img, m.astype(bool), (0.30, 0.28, 0.26), alpha=0.8, soft=0.002)
    found = detect(Patch(img, valid, MPP))
    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    f = _one(found, cx, cy, tol=0.15)
    assert f.damage_class == "structural_crack"
    true_len = np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    assert 0.7 * true_len < f.length[0] < 1.3 * true_len


def test_straight_axis_aligned_joint_is_not_a_crack():
    img, valid, _ = _wall()
    m = np.zeros((H, W), np.uint8)
    m[60:420, 300:302] = 1  # a panel joint
    _blend(img, m.astype(bool), (0.30, 0.28, 0.26), alpha=0.8, soft=0.002)
    assert detect(Patch(img, valid, MPP)) == []


def test_mould_soot_efflorescence_peeling_and_an_unnamed_class():
    img, valid, rng = _wall()
    # mould: dark green-black speckles in a cluster
    for _ in range(350):
        x, y = rng.normal((1.9, 1.9), 0.05)
        _blend(img, _ellipse(x, y, rng.uniform(0.008, 0.02), rng.uniform(0.008, 0.02)), (0.20, 0.24, 0.18), alpha=0.9, soft=0.002)
    # soot: large soft neutral dark patch
    _blend(img, _ellipse(0.9, 0.5, 0.22, 0.20), (0.35, 0.35, 0.36), alpha=0.6, soft=0.03)
    # efflorescence: light powdery speckle
    for _ in range(300):
        x, y = rng.normal((1.9, 0.6), 0.05)
        _blend(img, _ellipse(x, y, rng.uniform(0.003, 0.007), rng.uniform(0.003, 0.007)), (1.0, 1.0, 1.0), alpha=0.8, soft=0.002)
    # peeling paint: lighter irregular flakes with sharp edges
    for _ in range(12):
        x, y = rng.normal((1.4, 1.5), 0.06)
        poly = np.array([(x + rng.normal(0, 0.03) / MPP * MPP, y + rng.normal(0, 0.03)) for _ in range(5)])
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(poly / MPP).astype(np.int32)], 1)
        _blend(img, m.astype(bool), (0.98, 0.97, 0.93), alpha=1.0)
    # an unnamed class: a sharp saturated blue paint spill
    _blend(img, _ellipse(0.35, 1.0, 0.08, 0.06), (0.15, 0.30, 0.85), alpha=1.0, soft=0.002)
    found = detect(Patch(img, valid, MPP))
    got = {f.damage_class for f in found}
    assert "soot_or_fire" in got and "mould" in got
    assert _one(found, 1.9, 1.9, tol=0.1).damage_class == "mould"
    assert _one(found, 0.9, 0.5, tol=0.1).damage_class == "soot_or_fire"
    assert _one(found, 0.35, 1.0).damage_class == "other_anomaly"
    assert _one(found, 1.9, 0.6, tol=0.12).damage_class in ("efflorescence", "peeling_paint")  # a low wall, so salts are possible


def test_relief_pit_and_sag():
    img, valid, _ = _wall()
    relief = np.zeros((H, W), np.float32)
    relief[_ellipse(1.0, 1.0, 0.05, 0.05)] = -0.04  # a hole
    big = cv2.GaussianBlur(_ellipse(2.4, 1.3, 0.35, 0.3).astype(np.float32), (0, 0), 25) * 0.05  # a sag
    relief += big
    found = detect(Patch(img, valid, MPP, relief=relief))
    assert _one(found, 1.0, 1.0).damage_class == "hole_or_impact"
    assert _one(found, 2.4, 1.3, tol=0.15).damage_class == "sagging_or_bulging"


def test_occluded_area_is_never_damage_and_result_is_deterministic():
    img, valid, _ = _wall()
    a = detect(Patch(img, valid, MPP))
    b = detect(Patch(img, valid, MPP))
    assert a == [] and b == []
    _blend(img, _ellipse(1.6, 1.0, 0.12, 0.12), (0.72, 0.60, 0.40), alpha=0.55, soft=0.02)
    r1 = [(f.damage_class, f.width, f.height) for f in detect(Patch(img, valid, MPP))]
    r2 = [(f.damage_class, f.width, f.height) for f in detect(Patch(img, valid, MPP))]
    assert r1 == r2 and r1


def test_too_little_valid_surface_gives_nothing():
    img, valid, _ = _wall()
    valid[:] = False
    assert detect(Patch(img, valid, MPP)) == []


def test_objects_are_not_damage_but_a_large_stain_is():
    img, valid, _ = _wall()
    door = np.zeros((H, W), np.uint8)
    door[60:400, 300:480] = 1  # a door leaf: 0.9 m wide, 1.7 m high, dark, sharp, axis-aligned
    _blend(img, door.astype(bool), (0.45, 0.32, 0.20), alpha=1.0, soft=0.001)
    small = np.zeros((H, W), np.uint8)
    small[150:190, 200:270] = 1  # a light switch plate or a laptop: small, sharp, rectangular
    _blend(img, small.astype(bool), (0.95, 0.95, 0.95), alpha=1.0, soft=0.001)
    assert detect(Patch(img, valid, MPP)) == []
    img, valid, _ = _wall()
    _blend(img, _ellipse(1.6, 1.0, 0.45, 0.4), (0.72, 0.60, 0.40), alpha=0.55, soft=0.03)  # about 0.55 m2
    assert [f.damage_class for f in detect(Patch(img, valid, MPP))] == ["water_stain"]


def test_faint_ridges_are_texture_and_unclassified_regions_can_be_switched_off():
    img, valid, rng = _wall()
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (100, 100), (160, 130), 1, 1)
    _blend(img, m.astype(bool), (0.80, 0.78, 0.73), alpha=0.5)  # a barely darker scratch: lightness ~2 below
    assert detect(Patch(img, valid, MPP)) == []
    _blend(img, _ellipse(0.35, 1.0, 0.08, 0.06), (0.15, 0.30, 0.85), alpha=1.0, soft=0.002)
    assert [f.damage_class for f in detect(Patch(img, valid, MPP))] == ["other_anomaly"]
    assert detect(Patch(img, valid, MPP), report_unclassified=False) == []


def test_tilt_and_bow_of_the_plane_are_not_relief_damage():
    img, valid, _ = _wall()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    relief = 0.06 * (xx / W - 0.5) * 2 + 0.05 * ((yy / H - 0.5) * 2) ** 2  # a tilted, bowed plane fit: up to 6 cm off
    relief += np.random.default_rng(3).normal(0, 0.004, relief.shape).astype(np.float32)
    assert detect(Patch(img, valid, MPP, relief=relief)) == []


def test_a_protruding_fixture_glare_and_floor_brightness_are_not_damage():
    img, valid, _ = _wall()
    relief = np.zeros((H, W), np.float32)
    blob = _ellipse(1.0, 1.0, 0.10, 0.12)
    _blend(img, blob, (0.25, 0.25, 0.27), alpha=0.9, soft=0.004)  # a dark sink or tap
    relief[blob] = 0.05  # standing 5 cm off the wall
    assert detect(Patch(img, valid, MPP, relief=relief)) == []
    img, valid, _ = _wall()
    _blend(img, _ellipse(1.6, 1.8, 0.12, 0.10), (1.0, 1.0, 1.0), alpha=0.35, soft=0.01)  # glare high on a wall
    high = detect(Patch(img, valid, MPP))
    assert all(f.damage_class != "efflorescence" for f in high)
    img, valid, _ = _wall()
    _blend(img, _ellipse(1.6, 1.8, 0.12, 0.10), (1.0, 1.0, 1.0), alpha=0.35, soft=0.01)
    assert all(f.damage_class != "efflorescence" for f in detect(Patch(img, valid, MPP, kind="floor")))


def test_a_stain_on_paint_is_found_beside_a_large_door_of_another_material():
    """A wall a fifth of which is a grained brown door: the stain must not be lost in the door's colour spread."""
    img, valid, rng = _wall()
    door = np.zeros((H, W), bool)
    door[60:400, 300:480] = True
    grain = cv2.GaussianBlur(rng.normal(0, 0.05, (1, W)).astype(np.float32), (0, 0), 2.0)  # vertical stripes
    img[door] = (np.array([0.45, 0.32, 0.20]) * (1.0 + np.tile(grain, (H, 1))[..., None]))[door]
    _blend(img, _ellipse(2.85, 1.0, 0.13, 0.15), (0.72, 0.60, 0.40), alpha=0.55, soft=0.02)
    found = detect(Patch(img, valid, MPP))
    assert [f.damage_class for f in found] == ["water_stain"], [(f.damage_class, f.evidence[:60]) for f in found]
    assert _one(found, 2.85, 1.0).damage_class == "water_stain"
