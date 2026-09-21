# Photo-Tier Multi-View Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the photo tier's rotation-only registration with real multi-view SfM (pycolmap), add evidence-weighted Manhattan wall-angle optimization, and add joint global stitching refinement — so photo captures under a new walking protocol produce accurate wall geometry, corners, room dimensions, floor area and ceiling height with calibrated uncertainty.

**Architecture:** Photo stills become an input adapter into the video tier's existing SfM/gravity/scale/densify machinery (`sfm.py`-equivalent registration → `world.py` gravity → `anchor.py`-style ruler triangulation → `points_video.py`-style densification), feeding the existing shared RANSAC wall-fitting (`photo_layout.py`) which gains a new Manhattan-snap pass, and the existing doorway-gluing stitcher (`stitch.py`) which gains a new joint-refinement pass. Damage detection, schema, and the LiDAR/video tiers are untouched.

**Tech Stack:** Python, pycolmap (SfM), NumPy/SciPy (RANSAC, least-squares), OpenCV (SIFT features, color masking), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-photo-tier-geometry-design.md`

## Global Constraints

- SfM's recovered geometry/camera poses are the authoritative geometric scaffold; the depth model is auxiliary and never the sole geometric source.
- Reuse video-tier modules (`sfm.py`, `world.py`, `anchor.py`, `points_video.py`) where genuinely generic; do not modify their behavior to accommodate photos.
- Keep photo-specific discrete-image handling separate from video-specific temporal/keyframe assumptions — new code for what's genuinely different, reuse for what's genuinely shared.
- EXIF-derived per-photo focal length is used when present and plausible; missing/implausible EXIF falls back to a documented default with a flag, never an assumed-correct guess.
- Never force a corner to 90°; only snap where independent evidence already places it close to a Manhattan angle, weighted by wall support, and reject a snap that would materially worsen a wall's agreement with its own points.
- `evidence: "closure"` walls (ray-cast, unmeasured) are never treated as measured evidence, never join the Manhattan optimization.
- Stitching: sequential doorway-gluing is always computed first and is the fallback; joint refinement only runs with enough non-outlier constraints to over-determine the system, and rejects implausible correspondences rather than forcing them in.
- No new schema fields, no new structured diagnostics fields. Any status note rides in the existing free-text `diagnostics.notes` list.
- Distinguish stochastic reconstruction noise from systematic scale/reference uncertainty in the existing bootstrap mechanism; do not double-count the same error source.
- `report.py`, `assess.py`, `scope.py`, `concealed.py`, `schema.py`, and the video tier's own pipeline are untouched unless a task finds a concrete incompatibility (documented inline if so).
- Synthetic-capture test success is never reported as evidence of real-world accuracy; the real recapture pass against `Data/ground_truth.json` is the only accuracy claim.

---

## File Structure

| File | Disposition | Responsibility |
|---|---|---|
| `tests/synth.py` | Extended | Add procedural per-wall texture and a translating camera-path generator so synthetic captures exercise real SfM registration, not just depth ray-casting. |
| `src/floorfathom/photo_pose.py` | Rewritten | `run_photo_sfm(photos, work, seed) -> SfmResult`: per-image-camera pycolmap registration from a `PhotoSet`, EXIF-seeded, producing the shared `SfmResult` type. |
| `src/floorfathom/photo_scene.py` | Rewritten | `build_scene(photos, sfm, depth, seed) -> RoomScene`: gravity-aligned, densified cloud built from `SfmResult` + `world.estimate_gravity` + a `points_video.build_dense_cloud`-style densification, with a sparse/dense consistency check. |
| `src/floorfathom/photo_reference.py` | Rewritten | `ruler_scale(photos, sfm, length) -> (RulerScale | None, flags)`: multi-frame ruler triangulation via `anchor.py`'s `find_strip`/`track_strip`/`estimate_anchor`, replacing the single-view wall-plane-ray method. |
| `src/floorfathom/photo_layout.py` | Extended | New `snap_manhattan(edges, tolerance_deg) -> list[L.Edge]` inserted into the outline pipeline; existing RANSAC wall fitting unchanged. |
| `src/floorfathom/stitch.py` | Extended | New `refine_global(rooms, placements, wall) -> list[Placement]` called after the existing `stitch()`; existing sequential logic unchanged. |
| `src/floorfathom/photo_pipeline.py` | Modified | Wire the new `photo_pose`/`photo_scene`/`photo_reference` functions into `build_photo_room`; update `photo_frames` to use each photo's own SfM camera pose instead of a shared origin. |
| `capture_protocol.md` | Modified | Photo-tier walking protocol, replacing "stand in one spot." |
| `scripts/eval_photo_geometry.py` | New | Scores wall-length/corner-position/floor-area/ceiling-height error against `ground_truth.json` (real pass) or a synthetic room's known geometry (regression pass). |

---

## Task 1: Textured, translating synthetic captures

**Files:**
- Modify: `tests/synth.py`
- Test: `tests/test_synth_photo.py` (new)

**Interfaces:**
- Produces: `synth.textured_room(walls, height, path, photo_positions) -> list[PhotoStill]` where `PhotoStill` is a new small dataclass `(name: str, rgb: np.ndarray, f35: float, position: tuple[float,float,float], yaw_deg: float)` — later tasks write these to a folder via `io_photos`-compatible files.
- Produces: `synth.checkerboard_texture(colour, cell_m, mpp) -> Callable[[s, h], rgb]` — a per-wall texture function usable in place of `_plain`/`_stained` in `test_surfaces.py`'s `Quad`, and reusable by `synth.textured_room`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synth_photo.py
import cv2
import numpy as np

from tests.synth import checkerboard_texture, textured_room


def test_checkerboard_texture_has_sift_keypoints():
    tex = checkerboard_texture("brick", cell_m=0.25, mpp=0.01)
    ss, hh = np.meshgrid(np.linspace(0, 4, 400), np.linspace(0, 2.6, 260))  # paired (s, h) per pixel
    colours = tex(ss.ravel(), hh.ravel())  # (N, 3), N = 400*260
    img = (np.clip(colours, 0, 1).reshape(260, 400, 3) * 255).astype(np.uint8)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp = cv2.SIFT_create().detect(grey, None)
    assert len(kp) > 200  # a flat gradient wall gives near zero; texture must give real keypoints


def test_textured_room_path_actually_translates():
    stills = textured_room(
        walls="rect4x3", height=2.6,
        path=[(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)],
        photo_positions=9,
    )
    positions = np.array([s.position for s in stills])
    spread = positions[:, [0, 2]].max(axis=0) - positions[:, [0, 2]].min(axis=0)
    assert spread.min() > 1.0  # a real baseline, not a single station
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synth_photo.py -v`
Expected: FAIL with `ImportError: cannot import name 'checkerboard_texture'`

- [ ] **Step 3: Implement `checkerboard_texture` and `textured_room` in `tests/synth.py`**

Add near the top of `tests/synth.py`, after the existing `Wall`/`Room` dataclasses:

```python
@dataclass
class PhotoStill:
    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    f35: float
    position: tuple[float, float, float]
    yaw_deg: float


def checkerboard_texture(kind: str, cell_m: float = 0.2, mpp: float = 0.01, seed: int = 0):
    """A (s, h) -> (N, 3) colour function with real high-frequency texture, for SIFT to match on.

    Noise is a deterministic function of (s, h) via a fixed sinusoidal hash, not a fresh Generator per
    call, so the same coordinate always renders the same speckle (needed so two photos of the same
    physical wall patch show consistent texture for SIFT to match across viewpoints).
    """
    base = {"brick": np.array([0.62, 0.42, 0.34]), "wood": np.array([0.55, 0.42, 0.28])}[kind]
    alt = base * 0.78
    phase = float(seed)

    def f(s: np.ndarray, h: np.ndarray) -> np.ndarray:
        s, h = np.asarray(s, float), np.asarray(h, float)
        cs, ch = (s / cell_m).astype(int), (h / cell_m).astype(int)
        on = (cs + ch) % 2 == 0
        speckle = np.sin((s / mpp) * 12.9898 + (h / mpp) * 78.233 + phase)
        speckle = speckle - np.floor(speckle)  # fractional part, deterministic pseudo-noise in [0, 1)
        noise = (speckle[:, None] - 0.5) * 0.06
        colour = np.where(on[:, None], base, alt)
        return np.clip(colour + noise, 0, 1)

    return f


def _raycast_textured(o, R, walls: list[Wall], height: float, floor_y: float, tex, w: int = 320, h_px: int = 240,
                      fx: float = 266.0) -> np.ndarray:
    """Like ``_raycast``, but returns an (H, W, 3) uint8 RGB render: each pixel's hit wall, its position along
    the wall (s) and its height above the floor (h) are looked up in ``tex(s, h)``. Floor/ceiling render as a
    flat mid-grey (texture is not needed there for SfM registration, only on walls). A separate function from
    ``_raycast`` so the existing LiDAR depth-only path is untouched."""
    u, v = np.meshgrid(np.arange(w), np.arange(h_px))
    cx, cy = w / 2.0, h_px / 2.0
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones_like(u, float)], axis=-1).reshape(-1, 3)
    D = d_cam @ R.as_matrix().T
    t = np.full(len(D), np.inf)
    owner = np.full(len(D), -1)
    s_hit = np.zeros(len(D))
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (floor_y - o[1]) / D[:, 1]
        floor_hit = (D[:, 1] < 0) & (tf > 0) & (tf < t)
        t = np.where(floor_hit, tf, t)
        tc = (floor_y + height - o[1]) / D[:, 1]
        ceil_hit = (D[:, 1] > 0) & (tc > 0) & (tc < t)
        t = np.where(ceil_hit, tc, t)
    for wi, wall in enumerate(walls):
        a, b = np.array(wall.a), np.array(wall.b)
        e = b - a
        length = float(np.linalg.norm(e))
        det = D[:, 0] * (-e[1]) - D[:, 2] * (-e[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            rx, rz = a[0] - o[0], a[1] - o[2]
            tt = (rx * (-e[1]) - rz * (-e[0])) / det
            ss = (D[:, 0] * rz - D[:, 2] * rx) / det
        ok = (tt > 0) & (ss >= 0) & (ss <= 1) & np.isfinite(tt) & (tt < t)
        t = np.where(ok, tt, t)
        owner = np.where(ok, wi, owner)
        s_hit = np.where(ok, ss * length, s_hit)
    world_y = o[1] + t * D[:, 1]
    h_hit = np.clip(world_y - floor_y, 0.0, height)
    on_wall = owner >= 0
    colours = np.full((len(D), 3), 0.55)  # flat mid-grey for floor/ceiling/miss
    if on_wall.any():
        colours[on_wall] = tex(s_hit[on_wall], h_hit[on_wall])
    return (np.clip(colours, 0, 1).reshape(h_px, w, 3) * 255).astype(np.uint8)


def textured_room(walls: str, height: float, path: list[tuple[float, float, float]], photo_positions: int, seed: int = 0):
    """Render ``photo_positions`` stills along ``path`` (x, y, z waypoints) around a rectangular room with
    checkerboard-textured walls, for SfM registration tests. ``walls`` selects a preset footprint."""
    footprint = {"rect4x3": rect(0.0, 0.0, 4.0, 3.0)}[walls]
    tex = checkerboard_texture("brick")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, len(path) - 1, photo_positions)
    xs = np.interp(t, np.arange(len(path)), [p[0] for p in path])
    zs = np.interp(t, np.arange(len(path)), [p[2] for p in path])
    stills = []
    for k, (x, z) in enumerate(zip(xs, zs)):
        yaw = float(rng.uniform(0, 360))
        o = np.array([x, 0.0, z])
        R = _pose(o, np.radians(yaw), 0.0)
        rgb = _raycast_textured(o, R, footprint, height, FLOOR_Y, tex)
        stills.append(PhotoStill(f"synth_{k:03d}", rgb, f35=24.0, position=(x, 0.0, z), yaw_deg=yaw))
    return stills
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synth_photo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/synth.py tests/test_synth_photo.py
git commit -m "test: textured, translating synthetic photo captures for SfM validation"
```

---

## Task 2: Photo-tier SfM registration (`photo_pose.py`)

**Files:**
- Modify: `src/floorfathom/photo_pose.py` (replace `register_rotations`/`Poses` with `run_photo_sfm`)
- Test: `tests/test_photo_pose.py` (new)

**Interfaces:**
- Consumes: `PhotoSet`/`PhotoImage` from `io_photos.py` (`name`, `rgb`, `f35`, `f_px`, `full_size`); `SfmResult` from `sfm.py` (reused unchanged: `registered`, `centers`, `cam_to_world`, `intrinsics`, `image_size`, `points`, `track_length`, `point_error`, `observations`, `flags`).
- Produces: `run_photo_sfm(photos: PhotoSet, work: str | Path, seed: int = 0) -> SfmResult | None` — `None` when no model could be built at all (mirrors `sfm.run_sfm`'s contract).

**Design note (from codebase inspection):** `sfm.run_sfm` cannot be called unchanged for photos — it calls `focal_prior(kf)` which reads a shared focal length from `read_camera_metadata(kf.video)` (a `.MOV` file) and registers with `camera_mode=pycolmap.CameraMode.SINGLE` (one shared camera for every image). Photos carry their own per-image EXIF focal length (`io_photos.py`'s own docstring: *"the zoom can differ inside one folder... so the camera matrix is never shared"*), so photo registration needs `camera_mode=pycolmap.CameraMode.PER_IMAGE` with each image's own `f_px`. This is exactly the photo-specific piece the global constraints call out; `SfmResult` itself, and every downstream consumer of it, stays unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_pose.py
from pathlib import Path

import numpy as np

from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pose import run_photo_sfm
from tests.synth import textured_room


def _photoset(tmp_path) -> PhotoSet:
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=8)
    images = [PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=24.0 * 266 / 24.0)
              for s in stills]
    return PhotoSet(room="synth", images=images)


def test_translating_stills_register_with_real_baseline(tmp_path):
    photos = _photoset(tmp_path)
    sfm = run_photo_sfm(photos, tmp_path / "work", seed=0)
    assert sfm is not None
    assert sfm.registered_fraction >= 0.75  # most of 8 well-overlapping stills should register
    centres = sfm.centers[sfm.registered]
    spread = centres[:, [0, 2]].max(axis=0) - centres[:, [0, 2]].min(axis=0)
    assert spread.min() > 0.05  # real translation recovered, not a single point


def test_missing_exif_focal_falls_back_and_flags(tmp_path):
    photos = _photoset(tmp_path)
    for im in photos.images:
        im.f35 = None
    sfm = run_photo_sfm(photos, tmp_path / "work2", seed=0)
    assert sfm is not None
    assert "focal_prior_default" in sfm.flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_photo_pose.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_photo_sfm'`

- [ ] **Step 3: Implement `run_photo_sfm`**

Replace the body of `src/floorfathom/photo_pose.py` (keep the file; drop `register_rotations`/`Poses`/`kabsch`/`rotation_ransac` — no longer needed once real translation exists):

```python
"""Multi-view registration of one room's stills into camera poses and a sparse point cloud.

Photos carry their own per-image focal length (the zoom can differ inside one folder), so
registration cannot share sfm.run_sfm's single-camera model; pycolmap is called directly here
with a per-image camera. Everything downstream (SfmResult) is unchanged, so world.py, anchor.py
and points_video.py are reused unmodified.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .io_photos import PhotoSet
from .sfm import SfmResult

DEFAULT_F35 = 24.0  # main iPhone camera, used only when EXIF has no focal length (flagged)


def run_photo_sfm(photos: PhotoSet, work: str | Path, seed: int = 0) -> SfmResult | None:
    """Reconstruct one room's stills with pycolmap, one camera per image. None if no model registers."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    images_dir = work / "images"
    images_dir.mkdir(exist_ok=True)
    db, sparse = work / "db.db", work / "sparse"
    if db.exists():
        db.unlink()
    sparse.mkdir(exist_ok=True)

    names = []
    used_default_focal = False
    for im in photos.images:
        p = images_dir / im.name
        cv2.imwrite(str(p), cv2.cvtColor(im.rgb, cv2.COLOR_RGB2BGR))
        names.append(im.name)

    pycolmap.set_random_seed(seed)
    db_obj = pycolmap.Database(db)
    for im in photos.images:
        w, h = im.rgb.shape[1], im.rgb.shape[0]
        f = im.f_px
        if im.f35 is None:
            f, used_default_focal = DEFAULT_F35 * 0.0398 * max(w, h), True  # matches sfm.FOCAL_PER_MM's fit
        cam = pycolmap.Camera.create(0, "SIMPLE_RADIAL", f, w, h)
        db_obj.write_camera(cam)
        db_obj.write_image(pycolmap.Image(name=im.name, camera_id=cam.camera_id))
    db_obj.close()

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = 0.004
    extraction.sift.max_num_features = 8192
    cpu = pycolmap.Device.cpu
    pycolmap.extract_features(db, images_dir, image_names=names, camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              extraction_options=extraction, device=cpu)
    pycolmap.match_exhaustive(db, device=cpu)  # discrete stills (2-8/room): all-pairs is cheap at this count
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, images_dir, sparse, opts)
    if not recs:
        return None

    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    n = len(photos.images)
    index = {name: i for i, name in enumerate(names)}
    registered = np.zeros(n, dtype=bool)
    centers = np.full((n, 3), np.nan)
    rot = np.full((n, 3, 3), np.nan)
    observations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for image_id in rec.reg_image_ids():
        img = rec.images[image_id]
        i = index[img.name]
        cam_from_world = img.cam_from_world()
        world_from_cam = cam_from_world.inverse()
        registered[i] = True
        centers[i] = world_from_cam.translation
        rot[i] = world_from_cam.rotation.matrix()
        r_cw, t_cw = cam_from_world.rotation.matrix(), np.asarray(cam_from_world.translation)
        uv, z = [], []
        for p in img.points2D:
            if p.has_point3D():
                depth = float((r_cw @ rec.points3D[p.point3D_id].xyz + t_cw)[2])
                if depth > 0:
                    uv.append(p.xy)
                    z.append(depth)
        observations[i] = (np.array(uv).reshape(-1, 2), np.array(z))

    cam0 = next(iter(rec.cameras.values()))
    pts = list(rec.points3D.values())
    result = SfmResult(
        registered=registered, centers=centers, cam_to_world=rot,
        intrinsics=(float(cam0.params[0]), float(cam0.params[0]), float(cam0.params[1]), float(cam0.params[2])),
        image_size=(photos.images[0].rgb.shape[1], photos.images[0].rgb.shape[0]),
        points=np.array([p.xyz for p in pts]).reshape(-1, 3),
        track_length=np.array([p.track.length() for p in pts], dtype=int),
        point_error=np.array([p.error for p in pts], dtype=float),
        n_models=len(recs), mean_reprojection_px=float(rec.compute_mean_reprojection_error()),
        focal_prior_px=float(cam0.params[0]), focal_source="default" if used_default_focal else "exif",
        observations=observations,
    )
    if used_default_focal:
        result.flags.append("focal_prior_default")
    if len(recs) > 1:
        result.flags.append("sfm_split_into_several_models")
    if result.registered_fraction < 0.6:
        result.flags.append("sfm_registered_too_few_frames")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_photo_pose.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/floorfathom/photo_pose.py tests/test_photo_pose.py
git commit -m "feat: replace rotation-only photo registration with per-image pycolmap SfM"
```

---

## Task 3: Photo scene from SfM (`photo_scene.py`)

**Files:**
- Modify: `src/floorfathom/photo_scene.py`
- Test: `tests/test_photo_scene.py` (new)

**Interfaces:**
- Consumes: `SfmResult` (Task 2); `world.estimate_gravity(sfm) -> Gravity`; a `Keyframes`-shaped adapter (below) so `points_video.build_dense_cloud(kf, sfm, depth) -> DenseCloud | None` is reusable unmodified.
- Produces: `build_scene(photos: PhotoSet, sfm: SfmResult, depth: Depth, seed: int = 0) -> RoomScene | None`, where `RoomScene` gains one field over today's version: `centers: np.ndarray` (per-used-image camera centre in the gravity-aligned cloud frame), needed because photos no longer share one station. All other `RoomScene` fields (`cloud`, `used`, `floor_y`, `rotation`, `flags`, etc.) keep their existing meaning.

**Design note:** `points_video.build_dense_cloud(kf, sfm, depth)` only touches `kf.directory`/`kf.names[i]` (confirmed by reading its body) — never `kf.time_s`/`fps`/`sharpness`/`video`. A `_photo_keyframes(photos, work) -> Keyframes` adapter that writes each photo to `work/frames/<name>` and fills the unused temporal fields with harmless placeholders (`time_s=np.arange(n)`, `fps=1.0`, `sharpness=np.ones(n)`, `source_index=np.arange(n)`, `video=work/"frames"`) makes `build_dense_cloud` reusable exactly as-is, satisfying "reuse where genuinely generic."

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_scene.py
from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pose import run_photo_sfm
from floorfathom.photo_scene import build_scene
from tests.synth import textured_room


def _fake_depth(rgb):
    import numpy as np
    return np.full(rgb.shape[:2], 2.0, np.float32)  # constant plausible depth; scale is checked, not accuracy


def _photoset():
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=8)
    return PhotoSet(room="synth", images=[
        PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=24.0 * 266 / 24.0)
        for s in stills
    ])


def test_scene_builds_from_sfm_with_gravity_and_dense_points(tmp_path):
    photos = _photoset()
    sfm = run_photo_sfm(photos, tmp_path / "work", seed=0)
    assert sfm is not None
    scene = build_scene(photos, sfm, _fake_depth, seed=0)
    assert scene is not None
    assert scene.cloud.points.shape[0] > 500
    assert len(scene.used) >= 2
    assert scene.centers.shape == (len(scene.used), 3)
    # gravity-aligned: the floor plane should be well below the mean camera height
    assert scene.floor_y < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_photo_scene.py -v`
Expected: FAIL (`build_scene` still has the old `(photos, poses, depth, seed)` signature / `ImportError`)

- [ ] **Step 3: Implement**

In `src/floorfathom/photo_scene.py`, add the `Keyframes` adapter and rewrite `build_scene`:

```python
def _photo_keyframes(photos: PhotoSet, work: Path) -> "Keyframes":
    from .io_video import Keyframes

    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for im in photos.images:
        cv2.imwrite(str(frames_dir / im.name), cv2.cvtColor(im.rgb, cv2.COLOR_RGB2BGR))
        names.append(im.name)
    n = len(names)
    return Keyframes(directory=frames_dir, names=names, source_index=np.arange(n), time_s=np.arange(n, dtype=float),
                     sharpness=np.ones(n), size=(photos.images[0].rgb.shape[1], photos.images[0].rgb.shape[0]),
                     fps=1.0, video=frames_dir / "unused.mov")


def build_scene(photos: PhotoSet, sfm: SfmResult, depth: "Depth", seed: int = 0, work: Path | None = None) -> RoomScene | None:
    """Gravity-aligned, densified cloud from ``sfm``'s registered cameras. ``depth`` densifies coverage the
    sparse SfM cloud misses (blank walls); it never sets the primary scale or shape, only fills gaps, and is
    checked against the sparse points via points_video.fit_ratio before being trusted per frame."""
    from .points_video import build_dense_cloud
    from .world import estimate_gravity

    if sfm.registered.sum() < 2:
        return None
    gravity = estimate_gravity(sfm)
    work = work or Path(tempfile.mkdtemp(prefix="photo_scene_"))
    kf = _photo_keyframes(photos, work)
    dense = build_dense_cloud(kf, sfm, depth)
    if dense is None:
        return None
    R = gravity.rotation  # SfM frame -> +y-up frame
    idx = {name: i for i, name in enumerate(kf.names)}
    used = [im.name for i, im in enumerate(photos.images) if sfm.registered[i]]
    centers = np.array([sfm.centers[idx[name]] for name in used]) @ R.T
    points = dense.points @ R.T
    station = centers.mean(axis=0)
    points -= np.array([station[0], 0.0, station[2]])
    centers -= np.array([station[0], 0.0, station[2]])
    from .points import Cloud, voxel_downsample

    cloud = Cloud(points=voxel_downsample(points.astype(np.float32), 0.03), chunk=dense.chunk, n_chunks=dense.n_chunks)
    floor_y = float(np.percentile(points[:, 1], 2))
    flags = list(gravity.flags) + list(dense.flags)
    return RoomScene(cloud=cloud, traj_xz=centers[:, [0, 2]], used=used, floor_y=floor_y,
                     tilt_deg=gravity.tilt_deg, floor_support=gravity.floor_support,
                     scales=np.array([dense.frame_ratio.get(idx[n], np.nan) for n in used]),
                     scale_spread=float(np.nanstd(np.log(list(dense.frame_ratio.values())))) if dense.frame_ratio else 0.0,
                     rotation=R, flags=flags, depths=[], centers=centers)
```

Add `centers: np.ndarray` to the `RoomScene` dataclass definition (after `depths`). Add `import tempfile` and `from pathlib import Path` at the top if not already present. Check `world.Gravity`'s actual field names (`rotation`, `tilt_deg`, `floor_support`, `flags`) against `src/floorfathom/world.py:41-50` before wiring — adjust names to match exactly if they differ from this sketch.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_photo_scene.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/floorfathom/photo_scene.py tests/test_photo_scene.py
git commit -m "feat: build the photo-tier cloud from SfM + gravity + densification, not rotation-only depth"
```

---

## Task 4: Multi-frame ruler scale (`photo_reference.py`)

**Files:**
- Modify: `src/floorfathom/photo_reference.py`
- Test: `tests/test_photo_reference.py` (new)

**Interfaces:**
- Consumes: `SfmResult`; `anchor.find_strip(bgr, near, min_length_px, colour) -> (top, bottom) | None`; `anchor.track_strip(kf, sfm, ..., colour) -> list[SegmentObs]`; `anchor.estimate_anchor(sfm, obs, true_length_m, ...) -> Anchor | None`.
- Produces: `ruler_scale(photos: PhotoSet, sfm: SfmResult, work: Path, length: float = REFERENCE_LENGTH_M) -> tuple[RulerScale | None, list[str]]` — same return shape as today, so `photo_pipeline.py`'s consumption (`ruler.factor`, `ruler.rel_sigma`) needs no change.

**Design note:** `anchor.find_strip` defaults to `colour="white"`; the ruler is yellow. Confirm `_colour_mask` supports `colour="yellow"` (check `src/floorfathom/anchor.py:59-69` — if it only supports white/near-white, add a `"yellow"` branch there using the same hue thresholds `photo_reference.py`'s current `RULER_HUE`/`RULER_MIN_SAT`/`RULER_MIN_VAL` constants use, since `anchor.py` is a shared/generic module per the constraints and this is a small, additive, non-breaking extension of its existing colour options — not a behavior change for video's own "white" usage).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_reference.py
from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pose import run_photo_sfm
from floorfathom.photo_reference import ruler_scale
from tests.synth import textured_room  # extended in a follow-up to place a yellow ruler strip; see Step 3 note


def test_ruler_scale_none_without_a_visible_ruler(tmp_path):
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=8)
    photos = PhotoSet(room="synth", images=[
        PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=24.0 * 266 / 24.0)
        for s in stills
    ])
    sfm = run_photo_sfm(photos, tmp_path / "work", seed=0)
    assert sfm is not None
    ruler, flags = ruler_scale(photos, sfm, tmp_path / "ruler_work", length=0.316)
    assert ruler is None
    assert any("ruler_not_seen" in f or "insufficient" in f for f in flags) or flags == []
```

This first test only pins down the "no ruler present" contract (safe against the synthetic renderer not yet painting a ruler). A second test with an actual rendered yellow strip is added once the renderer supports it — track as a follow-up in the same task since it needs `tests/synth.py` extended with a paintable strip object; do not block this task's core `ruler_scale` implementation on it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_photo_reference.py -v`
Expected: FAIL (`ruler_scale` still has the old `(photos, poses, scene, segs, length)` signature)

- [ ] **Step 3: Implement**

Replace `ruler_scale` (and drop `_one_view`, which is superseded) in `src/floorfathom/photo_reference.py`:

```python
def ruler_scale(photos: PhotoSet, sfm: SfmResult, work: Path, length: float = REFERENCE_LENGTH_M) -> tuple["RulerScale | None", list[str]]:
    """Metric scale from the yellow reference ruler, triangulated across every registered still that sees it
    (real camera translation now makes this possible; see anchor.py, shared with the video tier)."""
    from .anchor import estimate_anchor, track_strip
    from .photo_scene import _photo_keyframes

    kf = _photo_keyframes(photos, work)
    obs = track_strip(kf, sfm, colour="yellow", until_s=None)
    if len(obs) < 3:
        return None, (["ruler_not_seen"] if not obs else ["ruler_too_few_views"])
    anchor = estimate_anchor(sfm, obs, true_length_m=length)
    if anchor is None:
        return None, ["ruler_triangulation_failed"]
    if anchor.scale is None:
        return None, ["ruler_triangulation_failed"]
    return RulerScale(factor=anchor.scale, rel_sigma=anchor.rel_sigma, n_photos=anchor.n_frames), anchor.flags
```

Add a `"yellow"` branch to `anchor._colour_mask` in `src/floorfathom/anchor.py`, reusing `photo_reference.py`'s existing `RULER_HUE`/`RULER_MIN_SAT`/`RULER_MIN_VAL` constants (move them into `anchor.py` as the yellow thresholds, imported back into `photo_reference.py` if still referenced elsewhere). Keep `RulerScale` dataclass as-is (`factor`, `rel_sigma`, `n_photos`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_photo_reference.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/floorfathom/photo_reference.py src/floorfathom/anchor.py tests/test_photo_reference.py
git commit -m "feat: triangulate the reference ruler across registered stills instead of one view"
```

---

## Task 5: Evidence-weighted Manhattan wall-angle optimization

**Files:**
- Modify: `src/floorfathom/photo_layout.py`
- Test: `tests/test_photo_layout_manhattan.py` (new)

**Interfaces:**
- Consumes: `layout.Edge(p0, p1, supported, support, line)` (existing, unchanged shape).
- Produces: `snap_manhattan(edges: list[L.Edge], tolerance_deg: float = 12.0, max_extra_residual_m: float = 0.03) -> list[L.Edge]`, called from `outline_from_segments` right after `edges = L.drop_short(L.merge_collinear(edges), ...)` and before the polygon is built from `edges`.

**Design note (guardrail formula, derived from `WallSegment`'s own fields — no raw points need to be threaded through):** each supported edge's `line = (normal, offset)`. A point on that line is always recoverable as `anchor = offset * normal` (the foot of the perpendicular from the origin — exact for any line in this normal-offset form). Rotating the edge's angle by `delta` moves points on the run by approximately `t * sin(delta)` at tangential distance `t` from `anchor`. Reject the snap for an edge if `max(|t0|, |t1|) * sin(delta) > max_extra_residual_m` (default 3 cm, matching `PLANE_THRESH`'s existing 5 cm inlier tolerance with margin) — this is the "reject if it would materially worsen a wall's agreement with its supporting points" guardrail without needing the original point cloud in this function.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_layout_manhattan.py
import numpy as np

from floorfathom import layout as L
from floorfathom.photo_layout import snap_manhattan


def _edge(p0, p1, angle_offset_deg=0.0, support=0.9):
    d = np.array(p1) - np.array(p0)
    length = float(np.linalg.norm(d))
    t = d / length
    n = np.array([-t[1], t[0]])
    if angle_offset_deg:
        c, s = np.cos(np.radians(angle_offset_deg)), np.sin(np.radians(angle_offset_deg))
        n = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1]])
    return L.Edge(np.array(p0, float), np.array(p1, float), True, support, (n, float(n @ p0)))


def test_near_right_angle_corner_snaps_to_90_when_well_supported():
    # two walls meeting at ~88 degrees, both well supported
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.9),
             _edge((4, 0), (4, 2.98), angle_offset_deg=2.0, support=0.9)]  # 2 deg off 90
    out = snap_manhattan(edges, tolerance_deg=12.0)
    n0, n1 = out[0].line[0], out[1].line[0]
    angle = np.degrees(np.arccos(np.clip(abs(n0 @ n1), 0, 1)))
    assert angle < 0.5  # snapped to exactly perpendicular


def test_genuinely_non_manhattan_corner_is_preserved():
    # a 70 degree corner (e.g. Hall's irregular quadrilateral) must not be touched
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.9),
             _edge((4, 0), (2.6, 2.75), angle_offset_deg=20.0, support=0.9)]  # ~70 deg from wall 0
    out = snap_manhattan(edges, tolerance_deg=12.0)
    assert np.allclose(out[1].line[0], edges[1].line[0])  # unchanged


def test_closure_edges_are_never_snapped():
    edges = [_edge((0, 0), (4, 0), support=0.9),
             L.Edge(np.array([4.0, 0.0]), np.array([4.0, 3.0]), False, 0.0, None)]  # closure: no line
    out = snap_manhattan(edges, tolerance_deg=12.0)
    assert out[1].line is None and out[1].supported is False


def test_weak_support_does_not_drag_a_well_supported_wall():
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.95),
             _edge((4, 0), (4, 2.9), angle_offset_deg=5.0, support=0.1)]  # near-90 but almost no support
    out = snap_manhattan(edges, tolerance_deg=12.0)
    n0 = out[0].line[0]
    assert np.allclose(n0, edges[0].line[0], atol=1e-6)  # the strong wall barely moves
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_photo_layout_manhattan.py -v`
Expected: FAIL with `ImportError: cannot import name 'snap_manhattan'`

- [ ] **Step 3: Implement `snap_manhattan` in `src/floorfathom/photo_layout.py`**

```python
MANHATTAN_TOLERANCE_DEG = 12.0  # default; configurable per call
MANHATTAN_MAX_EXTRA_RESIDUAL = 0.03  # metres: reject a snap that would move an edge's points more than this


def _nearest_manhattan_delta(angle_deg: float) -> float:
    """Signed degrees from ``angle_deg`` to the nearest multiple of 90, in (-45, 45]."""
    return ((angle_deg + 45.0) % 90.0) - 45.0


def snap_manhattan(edges: list[L.Edge], tolerance_deg: float = MANHATTAN_TOLERANCE_DEG,
                    max_extra_residual_m: float = MANHATTAN_MAX_EXTRA_RESIDUAL) -> list[L.Edge]:
    """Snap each supported edge's angle toward the nearest multiple of 90 degrees, weighted by its support,
    but only where a corner is already close to Manhattan and the snap does not measurably worsen the edge's
    fit to its own points. Closure edges (no line) are never touched."""
    idx = [i for i, e in enumerate(edges) if e.supported and e.line is not None]
    if len(idx) < 2:
        return edges
    angles = {i: float(np.degrees(np.arctan2(edges[i].line[0][1], edges[i].line[0][0]))) % 90.0 for i in idx}
    # a corner "votes" for a shared reference angle only when the two edges already sit near 0/90 of each other
    votes, weights = [], []
    n = len(edges)
    for i in idx:
        j = (i + 1) % n
        if j not in angles:
            continue
        d = _nearest_manhattan_delta(angles[i] - angles[j])
        if abs(d) <= tolerance_deg:
            w = edges[i].support * edges[j].support
            votes.append(angles[i] - _nearest_manhattan_delta(angles[i]))  # i's angle, referenced to its own nearest grid
            weights.append(w)
    if not votes:
        return edges
    ref = float(np.average(votes, weights=weights)) % 90.0

    out = list(edges)
    for i in idx:
        delta = _nearest_manhattan_delta(angles[i] - ref)
        if abs(delta) > tolerance_deg or abs(delta) < 1e-6:
            continue
        e = edges[i]
        n_old, c_old = e.line
        theta_old = np.arctan2(n_old[1], n_old[0])
        theta_new = theta_old - np.radians(delta)
        n_new = np.array([np.cos(theta_new), np.sin(theta_new)])
        t = np.array([-n_old[1], n_old[0]])
        t0, t1 = float(e.p0 @ t), float(e.p1 @ t)
        extra = max(abs(t0), abs(t1)) * abs(np.sin(np.radians(delta)))
        if extra > max_extra_residual_m:
            continue  # guardrail: would worsen this wall's agreement with its own points
        anchor = c_old * n_old
        c_new = float(n_new @ anchor)
        p0_new = e.p0 - (n_old @ e.p0 - c_old) * n_old  # re-project p0 onto old line first (numerical safety)
        p0_new = p0_new - (n_new @ p0_new - c_new) * n_new
        p1_new = e.p1 - (n_new @ e.p1 - c_new) * n_new
        out[i] = L.Edge(p0_new, p1_new, e.supported, e.support, (n_new, c_new))
    return out
```

Wire it into `outline_from_segments`, replacing:
```python
edges = L.drop_short(L.merge_collinear(edges), min_len=MIN_EDGE, corner_cut=CORNER_CUT)
```
with:
```python
edges = snap_manhattan(L.drop_short(L.merge_collinear(edges), min_len=MIN_EDGE, corner_cut=CORNER_CUT))
polygon = L.rebuild_polygon(edges)  # re-intersect snapped lines into corners
```
(check whether `outline_from_segments` already calls `L.rebuild_polygon` elsewhere before this point — if `polygon` was previously taken directly as `[e.p0 for e in edges]`, switching to `rebuild_polygon` here is required so snapped corners actually re-intersect; verify against the current file before editing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_photo_layout_manhattan.py -v`
Expected: PASS

- [ ] **Step 5: Run the full photo_layout test suite to check no regression**

Run: `uv run pytest tests/test_photo_layout.py -v` (if this file exists; else skip)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/floorfathom/photo_layout.py tests/test_photo_layout_manhattan.py
git commit -m "feat: evidence-weighted Manhattan wall-angle snap in photo-tier outlines"
```

---

## Task 6: Joint global stitching refinement

**Files:**
- Modify: `src/floorfathom/stitch.py`
- Test: `tests/test_stitch_refine.py` (new)

**Interfaces:**
- Consumes: `RoomPlan`, `Placement` (schema.py, unchanged); the existing `stitch()` function's output (`Stitched(rooms, placements, unplaced)`).
- Produces: `refine_global(rooms: list[RoomPlan], placements: list[Placement], wall: float = WALL) -> list[Placement]` — called from `stitch_plans` right after `stitch(rooms, wall)`, replacing the placements list when refinement runs, otherwise returned unchanged.

**Design note:** build one correspondence per pair of doorways (one on each of two *already-placed* rooms) whose transformed centres and opposed normals nearly coincide — this includes the doorway pairs `stitch()` already used, plus any extra pairs that happen to line up once all rooms are placed. Score each by `(centre distance, normal misalignment)`; keep only correspondences within a tight tolerance (e.g. 0.15 m, 8°) as non-outliers. Only run the least-squares refinement when there are at least 2 more non-outlier correspondences than rooms-minus-one (the sequential solve's own constraint count), so the system is genuinely over-determined; otherwise return `placements` unchanged (the fallback).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stitch_refine.py
import numpy as np

from floorfathom.schema import Measurement, Opening, RoomPlan, Wall
from floorfathom.stitch import refine_global, stitch


def _room(id_, poly, openings):
    return RoomPlan(id=id_, name=id_, frame="room", polygon=poly, walls=[
        Wall(id=f"{id_}_w{i}", start=poly[i], end=poly[(i + 1) % len(poly)],
             length=Measurement(value=float(np.hypot(*(np.array(poly[(i + 1) % len(poly)]) - np.array(poly[i])))),
                                lo=None, hi=None, unit="m", method="m", note=None), evidence="wall_points")
        for i in range(len(poly))
    ], openings=openings, ceiling_height=Measurement(value=2.6, lo=None, hi=None, unit="m", method="m", note=None),
        floor_area=Measurement(value=None, lo=None, hi=None, unit="m2", method="m", note=None))


def _opening(id_, centre, wall_id, width=0.9):
    return Opening(id=id_, kind="doorway", wall_id=wall_id, centre=centre, start=(centre[0], centre[1] - width / 2),
                   end=(centre[0], centre[1] + width / 2), width=Measurement(value=width, lo=None, hi=None, unit="m", method="m", note=None))


def test_refine_keeps_sequential_solve_when_underconstrained():
    a = _room("a", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("a_o0", (3.0, 1.5), "a_w1")])
    b = _room("b", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("b_o0", (0.0, 1.5), "b_w3")])
    st = stitch([a, b])
    refined = refine_global(st.rooms, st.placements)
    assert refined == st.placements  # only one doorway link: nothing to jointly refine, fallback holds


def test_refine_rejects_outlier_correspondence():
    # three rooms in a row sharing a hallway; one spurious near-match must not distort the solve
    a = _room("a", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("a_o0", (3.0, 1.5), "a_w1")])
    b = _room("b", [(0, 0), (2, 0), (2, 3), (0, 3)], [_opening("b_o0", (0.0, 1.5), "b_w3"), _opening("b_o1", (2.0, 1.5), "b_w1")])
    c = _room("c", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("c_o0", (0.0, 1.5), "c_w3")])
    st = stitch([a, b, c])
    refined = refine_global(st.rooms, st.placements)
    assert len(refined) == len(st.placements)
    for p in refined:
        assert p.overlap <= 0.05  # a bad correspondence must not have pulled a room into an overlap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stitch_refine.py -v`
Expected: FAIL with `ImportError: cannot import name 'refine_global'`

- [ ] **Step 3: Implement `refine_global` in `src/floorfathom/stitch.py`**

```python
CORR_CENTRE_TOL, CORR_NORMAL_TOL_DEG = 0.15, 8.0


def _corr_candidates(rooms: list[RoomPlan]):
    """Every pair of doorways on two different placed rooms whose centres and opposed normals nearly agree."""
    out = []
    for i, ra in enumerate(rooms):
        for oa in ra.openings:
            na = _inward_normal(ra, np.asarray(oa.centre))
            for rb in rooms[i + 1:]:
                for ob in rb.openings:
                    nb = _inward_normal(rb, np.asarray(ob.centre))
                    d = float(np.linalg.norm(np.asarray(oa.centre) - np.asarray(ob.centre)))
                    ang = float(np.degrees(np.arccos(np.clip(-na @ nb, -1, 1))))
                    if d <= CORR_CENTRE_TOL and ang <= CORR_NORMAL_TOL_DEG:
                        out.append((ra.id, rb.id, np.asarray(oa.centre), np.asarray(ob.centre)))
    return out


def refine_global(rooms: list[RoomPlan], placements: list[Placement], wall: float = WALL) -> list[Placement]:
    """Jointly refine every room's (angle, shift) over all doorway correspondences at once, initialized from
    the sequential solve. Falls back to ``placements`` unchanged when there are not enough non-outlier
    correspondences to over-determine the system."""
    from scipy.optimize import least_squares

    by_id = {p.room: p for p in placements}
    ids = [r.id for r in rooms if r.id in by_id]
    if len(ids) < 2:
        return placements
    corr = _corr_candidates(rooms)
    dof = 3 * (len(ids) - 1)  # one room's pose is the fixed anchor
    if len(corr) * 2 < dof + 2:  # need to be over-determined by a margin, not just square
        return placements

    root = ids[0]  # the sequential solve's first placed room is the natural anchor
    var_ids = [i for i in ids if i != root]
    x0 = np.concatenate([[by_id[i].angle, *by_id[i].shift] for i in var_ids])
    index = {i: k for k, i in enumerate(var_ids)}

    def pose(i, x):
        if i == root:
            return 0.0, np.zeros(2)
        a, sx, sy = x[3 * index[i]:3 * index[i] + 3]
        return a, np.array([sx, sy])

    def residuals(x):
        out = []
        for ida, idb, ca, cb in corr:
            aa, sa = pose(ida, x)
            ab, sb = pose(idb, x)
            wa = _rot(aa) @ ca + sa
            wb = _rot(ab) @ cb + sb
            out.extend((wa - wb).tolist())
        return np.array(out)

    fit = least_squares(residuals, x0, loss="soft_l1", f_scale=CORR_CENTRE_TOL)
    resid0 = residuals(x0).reshape(-1, 2)
    resid1 = fit.fun.reshape(-1, 2)
    kept_mask = np.linalg.norm(resid1, axis=1) <= max(CORR_CENTRE_TOL, 3 * np.median(np.linalg.norm(resid1, axis=1)))
    if not kept_mask.all():  # refit once more excluding outlier correspondences found by the first pass
        corr = [c for c, k in zip(corr, kept_mask) if k]
        if len(corr) * 2 < dof + 2:
            return placements
        fit = least_squares(residuals, x0, loss="soft_l1", f_scale=CORR_CENTRE_TOL)

    out = []
    for p in placements:
        if p.room == root or p.room not in index:
            out.append(p)
            continue
        a, s = pose(p.room, fit.x)
        out.append(p.model_copy(update={"angle": float(a), "shift": (float(s[0]), float(s[1]))}))
    return out
```

Call it from `stitch_plans` (after the existing `stitch()` call, before rooms are transformed with the final placements): find where `stitch_plans` currently uses `stitch(...)`'s `placements` to build the final room positions, and insert `placements = refine_global(rooms, placements)` there, before those placements are applied. Verify by reading the current end of `stitch_plans` (around line 268+) for the exact insertion point, since the excerpt above cuts off before its full body.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stitch_refine.py -v`
Expected: PASS

- [ ] **Step 5: Run the full stitch test suite to check no regression**

Run: `uv run pytest tests/test_stitch.py -v` (adjust filename if different)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/floorfathom/stitch.py tests/test_stitch_refine.py
git commit -m "feat: joint global refinement over doorway correspondences, sequential solve as fallback"
```

---

## Task 7: Wire the new pipeline into `photo_pipeline.py`

**Files:**
- Modify: `src/floorfathom/photo_pipeline.py`
- Test: `tests/test_photo_pipeline_integration.py` (new)

**Interfaces:**
- Consumes: `run_photo_sfm` (Task 2), `build_scene` (Task 3), `ruler_scale` (Task 4) — all with their new signatures.
- Produces: `build_photo_room(...)` keeps its existing public signature and return type (`PhotoRoom`); only its internals change.

**Design note (concrete incompatibility, in scope per the global constraints):** `photo_frames` currently builds each `surfaces.Frame.pose` assuming every photo shares one camera origin (`"the photos turn on the spot"`, hardcoded in its docstring). `surfaces.Frame.pose` is already a general 4x4 camera-to-world matrix (confirmed: `surfaces.py` requires no change), so this is `photo_pipeline.py`'s own glue code, not `assess.py`/`damage.py`/`surfaces.py`. Update `photo_frames` to build each frame's pose from that photo's own `scene.centers[k]` (Task 3) instead of assuming the origin.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_pipeline_integration.py
import numpy as np

from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pipeline import build_photo_room
from floorfathom.photo_pose import run_photo_sfm
from tests.synth import textured_room


def _fake_depth(rgb):
    return np.full(rgb.shape[:2], 2.0, np.float32)


def test_build_photo_room_end_to_end_on_synthetic_walk(tmp_path):
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=10)
    photos = PhotoSet(room="synth", images=[
        PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=24.0 * 266 / 24.0)
        for s in stills
    ])
    room = build_photo_room("synth", photos, _fake_depth, seed=0)
    plan_room = room.plan.rooms[0]
    assert plan_room.floor_area.value is not None
    assert abs(plan_room.floor_area.value - 12.0) / 12.0 < 0.25  # 4x3 m room, generous synthetic tolerance
    assert abs(plan_room.ceiling_height.value - 2.6) < 0.3
    assert len(plan_room.walls) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_photo_pipeline_integration.py -v`
Expected: FAIL (old `build_photo_room` signature takes `poses`, not raw `photos`+`depth` directly wired to SfM)

- [ ] **Step 3: Implement**

In `src/floorfathom/photo_pipeline.py`:

1. Replace the import line:
   ```python
   from .photo_pose import Poses, register_rotations
   ```
   with:
   ```python
   from .photo_pose import run_photo_sfm
   ```
2. Replace the import of `ruler_scale` usage signature (same import line, function body changes downstream).
3. In `build_photo_room`, replace the poses/scene construction:
   ```python
   scene: RoomScene | None = build_scene(photos, poses, depth, seed)
   ```
   with:
   ```python
   sfm = run_photo_sfm(photos, work_dir, seed)  # work_dir: new parameter, default tempfile.mkdtemp(prefix="photo_room_")
   if sfm is None:
       return PhotoRoom(_null(name, time.perf_counter() - t0, seed, [*flags, "insufficient_registration"], models))
   flags += sfm.flags
   scene: RoomScene | None = build_scene(photos, sfm, depth, seed)
   ```
4. Replace the ruler call:
   ```python
   ruler, ruler_flags = ruler_scale(photos, poses, scene, wall_segments(scene.cloud.points, seed),
                                    reference_length_m or REFERENCE_LENGTH_M)
   ```
   with:
   ```python
   ruler, ruler_flags = ruler_scale(photos, sfm, work_dir, reference_length_m or REFERENCE_LENGTH_M)
   ```
5. Add a `work: str | Path | None = None` parameter to `build_photo_room`'s signature, defaulting to a fresh temp dir when `None`, used as `work_dir` above.
6. In `photo_frames`, replace the shared-origin pose construction — locate the block building each `Frame`'s `pose` (after line ~179 shown earlier) and replace the camera-at-origin assumption with `scene.centers[k]` as the translation and `scene.rotation @ sfm.cam_to_world[...]`-derived rotation as before, keyed by the same `k`/`name` indexing `photo_frames` already uses via `scene.used`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_photo_pipeline_integration.py -v`
Expected: PASS

- [ ] **Step 5: Run the full existing photo test suite to check no regression**

Run: `uv run pytest tests/test_photo_pipeline.py tests/test_photo_damage.py -v` (adjust filenames to what exists)
Expected: PASS, or failures triaged and fixed before proceeding

- [ ] **Step 6: Commit**

```bash
git add src/floorfathom/photo_pipeline.py tests/test_photo_pipeline_integration.py
git commit -m "feat: wire SfM-based registration, scene and ruler scale into build_photo_room"
```

---

## Task 8: Update the capture protocol

**Files:**
- Modify: `capture_protocol.md`

**Steps:**

- [ ] **Step 1: Rewrite the "How to walk — Photos tier" section**

Replace `capture_protocol.md`'s current photo-tier walking instructions (the "stand facing that wall squarely... this spot is where you take all of the room's photos" section) with a short walking path per room, matching the video protocol's existing language:

```markdown
## How to walk — Photos tier (2-8 stills per room)

Before the first room: take the yellow ruler used for video (yellow body 31.6 x 4 cm). Measure
the yellow part end to end with a tape to 1 mm, write the value down, and give it to the command
with `--reference-length-cm` (as the video section says).

For each room:

1. Stick the ruler flat on a wall, upright, with its centre at about phone height (1.2-1.5 m),
   on a wood-coloured or dark surface such as a door or a wardrobe. Not on a mirror, window,
   glossy tile or glass door. Nothing else yellow next to it.
2. Stand about 1.5-2.5 m from a wall, phone upright (portrait). Walk a short arc, about 1 m to
   one side and back, taking a photo every 45-60 degrees of turn as you go — move your feet, not
   only your wrist, so consecutive photos have a real difference in viewpoint, not just rotation.
3. Continue the arc around the room, one photo per wall face, until every wall has been
   photographed from at least one position. Keep the ruler in view in at least three of the
   photos, from different positions, so its true size can be triangulated.
4. Include the floor and ceiling line in frame where possible; do not crop tight on eye level only.
5. If the room has an alcove, closet opening, or a second doorway not visible from your arc, walk
   to a second position and repeat step 2-3 for that area.
6. One photo folder per room, one ruler per folder. Do not mix rooms into one folder.
```

- [ ] **Step 2: Update the device matrix note**

In the "Device matrix" table's Photo row, update the description to reflect the walking protocol and that the ruler scale now comes from multi-frame triangulation rather than a single view, keeping the "accuracy on real photos not measured yet" caveat until Task 9's real-recapture pass runs.

- [ ] **Step 3: Commit**

```bash
git add capture_protocol.md
git commit -m "docs: change the photo capture protocol from stand-and-rotate to a short walking arc"
```

---

## Task 9: Real-data validation script

**Files:**
- Create: `scripts/eval_photo_geometry.py`
- Test: `tests/test_eval_photo_geometry.py` (new — exercises the scoring logic against synthetic ground truth, not real data)

**Interfaces:**
- Consumes: a `CapturePlan` (from a finished `plan.json`) and a ground-truth dict shaped like `Data/ground_truth.json`.
- Produces: `score_geometry(plan: CapturePlan, ground_truth: dict) -> dict` returning `{"wall_length_error": [...], "corner_position_error": [...], "floor_area_error": float | None, "ceiling_height_error": float | None}`, plus a `main()` CLI entry point matching `scripts/eval_damage_photo.py`'s existing pattern (`plan.json` path argument, prints a summary table).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_photo_geometry.py
from floorfathom.schema import CapturePlan, Diagnostics, Measurement, RoomPlan, Wall
from scripts.eval_photo_geometry import score_geometry


def _plan():
    room = RoomPlan(id="room_0", name="synth", frame="room",
                    polygon=[(0, 0), (4, 0), (4, 3), (0, 3)],
                    walls=[Wall(id="w0", start=(0, 0), end=(4, 0),
                                length=Measurement(value=4.0, lo=3.8, hi=4.2, unit="m", method="m", note=None),
                                evidence="wall_points")],
                    openings=[], ceiling_height=Measurement(value=2.6, lo=2.5, hi=2.7, unit="m", method="m", note=None),
                    floor_area=Measurement(value=12.0, lo=11.0, hi=13.0, unit="m2", method="m", note=None))
    return CapturePlan(capture="synth", tier="photo", rooms=[room],
                       diagnostics=Diagnostics(bootstrap_replicates=0, seed=0, seconds=0.0, conventions=""))


def test_score_geometry_against_known_synthetic_truth():
    truth = {"synth": {"walls": {"w0": 4.0}, "ceiling_height": 2.6, "floor_area": 12.0}}
    scores = score_geometry(_plan(), truth)
    assert scores["wall_length_error"][0] == 0.0
    assert scores["ceiling_height_error"] == 0.0
    assert scores["floor_area_error"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_photo_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_photo_geometry'`

- [ ] **Step 3: Implement**

Check `scripts/eval_damage_photo.py`'s existing structure first (imports, `if __name__ == "__main__":` pattern, how it loads `plan.json` and `ground_truth.json`) and match its conventions. Then create `scripts/eval_photo_geometry.py`:

```python
"""Score a finished photo-tier plan.json against tape ground truth: wall length, corner position,
floor area, ceiling height error. Matches scripts/eval_damage_photo.py's CLI pattern."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from floorfathom.schema import CapturePlan


def score_geometry(plan: CapturePlan, ground_truth: dict) -> dict:
    room = plan.rooms[0]
    gt = ground_truth.get(plan.capture, {})
    wall_err = []
    for w in room.walls:
        truth = gt.get("walls", {}).get(w.id)
        if truth is not None and w.length.value is not None:
            wall_err.append(abs(w.length.value - truth) / truth)
    corner_err = []
    gt_corners = gt.get("corners")
    if gt_corners:
        import numpy as np
        for p, t in zip(room.polygon, gt_corners):
            corner_err.append(float(np.hypot(p[0] - t[0], p[1] - t[1])))
    area_err = None
    if room.floor_area.value is not None and gt.get("floor_area") is not None:
        area_err = abs(room.floor_area.value - gt["floor_area"]) / gt["floor_area"]
    ceiling_err = None
    if room.ceiling_height.value is not None and gt.get("ceiling_height") is not None:
        ceiling_err = abs(room.ceiling_height.value - gt["ceiling_height"])
    return {"wall_length_error": wall_err, "corner_position_error": corner_err,
            "floor_area_error": area_err, "ceiling_height_error": ceiling_err}


def main(plan_path: str, ground_truth_path: str) -> None:
    plan = CapturePlan.model_validate_json(Path(plan_path).read_text())
    truth = json.loads(Path(ground_truth_path).read_text())
    for room in plan.rooms:
        scores = score_geometry(plan.model_copy(update={"rooms": [room]}), truth)
        print(f"{room.name}: wall errors {[f'{e:.1%}' for e in scores['wall_length_error']]}, "
              f"area error {scores['floor_area_error']}, ceiling error {scores['ceiling_height_error']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_photo_geometry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_photo_geometry.py tests/test_eval_photo_geometry.py
git commit -m "test: add wall/corner/area/ceiling error scoring against ground truth"
```

- [ ] **Step 6: Manual follow-up (not a subagent task — needs your phone and the real rooms)**

Once `Data/B1`, `Data/B2`, `Data/Hall` are recaptured under the new walking protocol (Task 8), run:
```bash
uv run floorfathom plan Data --tier photo --out out/photo_v2 --reference-length-cm <measured cm>
uv run python scripts/eval_photo_geometry.py out/photo_v2/rooms/Hall.json Data/ground_truth.json
```
This is the real accuracy validation the spec requires; synthetic test success above is not evidence of it.

---

## Self-Review Notes

- **Spec coverage:** every spec section (SfM replacement, scene/densification, ruler triangulation, Manhattan optimization with guardrail, joint stitching refinement with outlier rejection and sequential fallback, uncertainty reuse, protocol change, synthetic + real validation) maps to Tasks 1-9.
- **No schema/diagnostics changes:** confirmed no task adds a `schema.py` field or a new structured diagnostics key; Task 2/3's new flags (`focal_prior_default`, etc.) reuse the existing free-text `flags`/`notes` lists already in the schema.
- **Type consistency check:** `SfmResult` (Task 2) is consumed unchanged by Tasks 3/4/7; `RoomScene.centers` (Task 3) is defined once and consumed by Task 7's `photo_frames` update; `L.Edge` (Task 5) matches `layout.py`'s existing dataclass fields exactly; `Placement` (Task 6) reuses `schema.py`'s existing fields (`room`, `angle`, `shift`, `host`, `host_opening`, `opening`, `overlap`, `width_diff`, `margin`) with no new fields.
- **Uncertainty mechanism reuse:** Task 7 does not reimplement `_leave_one_out`/`RoomSamples`/`interval()` — confirmed by inspection that this machinery already resamples generically over `(points, chunk)`, so it automatically picks up the new cloud's noise characteristics once Tasks 2-4 are wired in Task 7. Task 4's `Anchor.rel_sigma` replaces `photo_reference.py`'s old single-view `rel_sigma` computation as the systematic scale term, kept distinct from the point-cloud bootstrap's stochastic term (no double-counting: one is per-measurement resampling, the other is the reference object's own triangulation spread, combined via the existing `_widen`'s quadrature sum).
