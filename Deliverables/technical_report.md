# FloorFathom — Technical Report

---

## 1. Architecture

FloorFathom is a three-tier pipeline (LiDAR, video, photo) that produces the same output
contract from three different sensor configurations. Every tier adapts its raw sensor inputs into a metric point cloud in a gravity-aligned frame,
then feeds it into a shared geometric core. The point cloud is the primary substrate for all
tiers; video additionally supplies camera-ray free-space votes (filling cells the glossy floor
starves of points) and globally-fitted wall lines as geometric hints to the shared estimator.
Photo supplies evidence-weighted Manhattan angle corrections after RANSAC wall fitting. LiDAR
supplies nothing extra — its depth is already metric and its cloud is dense enough for the
estimator to work unaided.

**Shared core modules** (all tiers):

| Module | Role |
|---|---|
| `estimate.py` | Room finder: grid → free space → polygon → walls → openings |
| `layout.py` | Occupancy grid, free-space carving, polygon simplification, wall snapping |
| `ransac.py` | Robust plane fitting (random triples, inlier refit, iterative refinement) |
| `planes.py` | Floor/ceiling detection via 1 cm height histogram peaks |
| `uncertainty.py` | Delete-d jackknife bootstrap + systematic terms → 95% intervals |
| `stitch.py` | Multi-room placement: doorway-gluing BFS + joint least-squares refinement |
| `damage.py / assess.py` | Surface unrolling, colour/crack/relief detection, region extraction |
| `concealed.py / scope.py` | Concealed-damage flag rules; repair scope items keyed to surfaces |
| `report.py / schema.py` | Assemble `CapturePlan` (Pydantic, schema 0.4.0); JSON output |
| `render.py` | Top-down PNG with dimensions, doorways, damage markers |

**Tier-specific adaptation modules:**

| Module | Tier | Role |
|---|---|---|
| `io_lidar.py` | LiDAR | Stray Scanner folder → depth + odometry + RGB |
| `drift.py` | LiDAR | Pose-graph drift correction (wall registration, weighted least-squares) |
| `lidar_frames.py` | LiDAR | Posed RGB+depth frames for damage |
| `io_video.py / sfm.py` | Video + Photo | Keyframe extraction; pycolmap SfM |
| `world.py` | Video + Photo | Gravity alignment from camera-up and horizontal layering |
| `anchor.py` | Video + Photo | Yellow ruler detection and metric scale triangulation |
| `depth.py / points_video.py` | Video + Photo | Depth Anything V2 densification fitted to SfM sparse cloud |
| `video_rays.py` | Video | Camera-ray free-space carving: open cells from visibility votes |
| `video_walls.py` | Video | Global vertical plane fitting to suppress inter-frame fragments |
| `video_heights.py` | Video | Density-based floor/ceiling plateau: more robust to depth-model bow |
| `video_openings.py` | Video | Ray-trace-based doorway detection |
| `photo_pose.py` | Photo | Per-image pycolmap SfM (one camera per still, EXIF focal length) |
| `photo_scene.py` | Photo | Gravity-aligned dense cloud from SfM + `points_video` densification |
| `photo_reference.py` | Photo | Multi-frame ruler triangulation via `anchor.py` with yellow strip |
| `photo_layout.py` | Photo | Evidence-weighted Manhattan wall-angle snap after RANSAC fitting |

The key insight is that photo and video tiers share the SfM, gravity, depth, and anchor modules —
they differ only in camera model (per-image vs shared) and the additional geometry hints
each tier can contribute (ray carving for video, Manhattan snap for photo).

---

## 2. Tier Design and Device Matrix

All tiers produce: per-room `walls`, `ceiling_height`, `floor_area`, `openings`, `damage`,
`concealed_flags`, `scope`, confidence interval on every measurement, `plan.json` to
schema 0.4.0, rendered `plan.png`. One command per capture.

**Sensor adaptation by tier:**

**LiDAR:**
Input is a Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`,
`camera_matrix.csv`). Depth is metric and absolute; no scale estimation needed. The walk
covers the whole property in one recording; drift is corrected by wall registration (§3).
The room estimator receives the voxel-downsampled cloud directly.

**Video:**
Input is one MOV clip per room from the iPhone native camera. Keyframes are extracted
every 0.2 s (sharpest-per-window). pycolmap SfM with a shared camera and EXIF-seeded
focal length recovers camera poses. Depth Anything V2 Metric-Indoor densifies each
frame's cloud fitted to the SfM sparse points by per-frame median ratio. Metric scale
comes from the yellow ruler (triangulated across ≥3 frames by `anchor.py`); if not found,
the depth model's own scale is used and the interval is widened by ≥15%. Three video-only
hints feed the shared estimator: ray free-space votes (open cells a camera ray crossed
without hitting a surface), global vertical-plane wall lines (suppressing per-chunk
fragments), and plateau-based floor/ceiling re-centring.

**Photo (post-SfM rewrite):**
Input is 2-8 stills per room from any iPhone 15+. pycolmap SfM with one camera per image
(EXIF focal length, `PER_IMAGE` camera mode) recovers poses with a real translation
baseline from the walking arc. The same `points_video` densification, `world.estimate_gravity`
gravity alignment, and `anchor.py` ruler triangulation as the video tier are reused without
modification. An evidence-weighted Manhattan snap is applied to wall angles before polygon
construction; `evidence="closure"` walls (ray-cast gaps) are never snapped.

**Device matrix:**

| Tier | Tested hardware | Scale source | Accuracy gate | Notes |
|---|---|---|---|---|
| Photo | iPhone 16 (non-Pro) | Yellow ruler (multi-frame triangulation) or depth model | Wall ±8%, ceiling ±1.5 cm | Ruler triangulated across ≥3 frames; without ruler, rel_σ ≥ 30%, rooms flagged |
| Video | iPhone 16 (non-Pro) | Yellow ruler or depth model | Wall ±3%, ceiling ±1.5 cm | 0.6× ultra-wide; ruler found in first ~8 s of clip |
| LiDAR | *(none — Cozmo sample data)* | Metric depth (absolute) | Wall ±3%, ceiling ±1.5 cm | No Pro device; accuracy unmeasured against tape |

---

## 3. Drift Handling

**LiDAR** (`drift.py`): Odometry from the IMU integrates (dx, dz, yaw) per frame; errors
accumulate over a multi-room walk. The correction: extract wall cells from the cloud (cells
occupied across ≥5 height bands, indicating a vertical surface), register overlapping chunk
pairs point-to-line (ICP-style in plan view), and solve for a (dx, dz, yaw) correction per
chunk by weighted least-squares with chunk 0 as a fixed reference. A weak prior penalises
large shifts between neighbouring chunks (≤0.05 m, ≤0.5°). Outlier pairs are rejected when
fewer than 25% of one chunk's wall cells match.

Ablation output is written to `plan.json` under `stitching.drift`: footprint area and room
count with correction ON and OFF, max shift (m), max yaw (°), χ² before and after. On the
Cozmo sample data, the correction was small (short walks, low odometry drift); the ablation
is still required by the gate and is always computed.

**Video:** Per-clip SfM is internally consistent (no accumulated drift within a clip). The
inter-clip stitcher (`stitch.py`) handles placement drift: doorway-gluing BFS gives an
initial placement, then `refine_global` runs a joint least-squares refinement over all
doorway correspondences simultaneously (soft-L1 loss, outlier rejection, falls back to
the sequential solve when the system is not over-determined). This is not pose-graph
drift correction — it is a global alignment of independent room frames.

**Photo:** Same stitching refinement as video. No intra-room drift (rotation-only walk in
one spot did not accumulate pose error; SfM with a walking arc is self-consistent).

---

## 4. Error Budget and Calibration Analysis

**Scale (dominant source):**
The depth model (Depth Anything V2 Metric-Indoor Small/Large) has a per-frame scale error
of ±15–30% on indoor rooms outside its training distribution. The yellow ruler reduces this
to ±3–5% (jackknife `rel_sigma` across triangulated frames). Every measurement's interval
includes scale uncertainty via the existing `_widen` mechanism (quadrature sum of
jackknife sampling spread and ruler `rel_sigma`); rooms without a ruler carry a wide flat
interval and a flag.

**SfM noise:**
Reprojection error is typically <1 px on registered frames. Per-frame depth ratio fitting
uses MAD-based outlier rejection; the spread of ratios across frames feeds the scale
systematic term. Photo SfM (per-image cameras, all-pairs matching) is stable for 2-8 stills
with real texture; plain/white walls cause registration failure (returned as `None`, room
falls back to null geometry with flags).

**Room estimator:**
Grid quantisation is 5 cm per cell. Polygon simplification uses a 12 cm tolerance (LiDAR)
or 25 cm (video) Douglas-Peucker step followed by corner snapping to wall points. The 25 cm
video tolerance was chosen because the depth-model point cloud bows and kinks by several cm
per wall; a tighter tolerance retains that noise as polygon vertices. Photo uses 12 cm
with an additional Manhattan snap that snaps wall angles toward the nearest 90° multiple
only when the snap moves the edge's own points by ≤3 cm.

**Ceiling height:**
Histogram mode on 1 cm bins (LiDAR, photo). Video uses `video_heights`: a kernel-smoothed
density histogram on points within a window of the histogram guess, taking the mode of the
smoothed distribution and computing the centre as the median of nearby points. The spread
(1.4826 × MAD) is fed back to `_widen` the ceiling interval. This is more robust to
floating outliers from the balcony glass or glossy floor reflections that shift the
histogram mode.

**Interval construction:**
1. Delete-d jackknife: N contiguous chunks, leave 2 out, refit, match rooms; robust spread
   = `1.4826 × MAD` across replicates, scaled by `sqrt((n−drop)/drop)`.
2. Systematic terms (hand-set, not calibrated from real data):
   walls ±1.5 cm + 0.5%, ceiling ±1.5 cm + 0.3%, area ±5 cm² + 1%, openings ±3 cm + 1%.
3. Combined: `half = sqrt((1.96 × jackknife_sd)² + systematic²)`.

**Calibration:** Only 4 rooms exist in `Data/` (the tape benchmark). Intervals are
structurally sound but empirically uncalibrated — systematic terms were chosen by
engineering judgement, not fitted to coverage data. On the known results (video H1
ceiling, photo Hall+B2 ceiling), the tape value fell inside the interval, suggesting
intervals are conservative. Wall gate failures indicate a systematic bias that exceeds
the assumed 0.5% + 1.5 cm term; the photo SfM rewrite and video wall aggregation
improvements are intended to reduce the bias, after which a recalibration pass against
the full benchmark is needed.

---

## 5. Fix Loop Story

*[PLACEHOLDER — to be completed after the video and photo tier changes land and the final
benchmark run establishes which gate is the true worst performer. Structure:*

*1. Gate name, failing number (measured vs tape)*
*2. Root-cause hypothesis and supporting evidence*
*3. Fix shipped (code diff, before/after reproduced run)*
*4. After number and whether the gate passed]*

---

## 6. Known Failure Modes

**All tiers:**
- *Scale bottleneck.* All geometry inherits the ruler-or-depth-model scale error. A 5% scale
  error moves a 5 m wall by 25 cm. Without a ruler the depth model can be off by 30%+.
- *Mirrors and glass.* Depth is wrong on reflective surfaces and see-through glass
  (balcony sliding door). The balcony wall-line filter in `video_walls` cuts off points
  behind a fitted wall plane, partially mitigating glass bleed-through.
- *Damage false positives.* Rules tuned on synthetic surfaces; real runs produce false
  positives on door frames, wood grain, door hardware, wardrobe edges. A CLIP classifier
  was evaluated and dropped (worse than rules on real patches).
- *Four-room benchmark.* The jackknife has too few rooms to calibrate intervals; reported
  widths are structurally correct but not empirically validated for coverage.

**LiDAR:**
- No live capture (no Pro device). Tier run on Cozmo-provided sample data; accuracy
  against tape is unmeasured. Damage regions on sample data are all false positives
  (no real damage exists in the sample scans).
- ~25% of walls pass the ±3% repeatability gate on sample data; root cause is the
  single-scan geometry (no second capture of the same room for comparison).

**Video:**
- Glossy tile floor (hall_kitchen): few floor-level depth points; the room polygon
  fragmented before ray free-space carving. With rays, the room is correctly connected
  but wall lengths still depend on the depth-model point cloud.
- Run-to-run variation: keyframe selection and SfM are not fully deterministic between
  runs on the same clip; wall lengths and damage regions can shift.
- B1 clip: stays in its own frame when no doorway observation links it to H1 (stitching
  requires a visible doorway crossing).

**Photo (pre-SfM rewrite):**
- Rotation-only registration gave no baseline; depth scale inconsistent across photos;
  floor area null in all rooms (outline incomplete); ceiling errors of +31 cm and -35 cm.

**Photo (post-SfM rewrite):**
- Real SfM requires ≥2 frames with a real translation baseline and overlapping texture.
  Fewer than 60% registration rate → `sfm_registered_too_few_frames` flag, room may
  fall back to null geometry.
- Plain or white walls give few SIFT keypoints and poor registration.
- Without a ruler, scale falls back to the depth model and intervals widen to ±30%+.
