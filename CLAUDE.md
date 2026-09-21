# FloorFathom

Cozmo AI case study (`Applied AI.pdf`): from handheld iPhone captures (photos, video,
LiDAR) produce a dimensioned, stitched whole-property floor plan plus damage regions
and scope items, one command per capture, JSON to the published schema. Scoring:
walk-in test 30%, fix loop 25%, benchmark accuracy 15%, compliance matrix 10%,
head-to-head vs a consumer app 10%, capture route 5%, process evidence 5%.

## Docs

- `capture_protocol.md`: the required one-page capture protocol, followed literally
  at the defense. Keep it in sync with how the data was actually captured.
- `edge_cases.md`: edge cases in the capture data the pipeline must handle.
- Both files state what we have, not every step taken.

## Data

- `Data/`: iPhone 16 captures (no LiDAR device). Photos are HEIC in `images/<room>/`,
  videos are `.MOV` at the top level. The LiDAR tier uses Cozmo-provided sample data.
- Ground truth is measured using tape. Staged damage and real damage are labeled separately.


## Code

- `uv sync`, then `uv run floorfathom plan <capture_folder> --out out/<name>`; tests with `uv run pytest`
  (about 6 minutes, synthetic captures in `tests/synth.py`).
- Every tier, its command and its state are in "Tiers and commands" below. See `README.md` for the module map,
  verification numbers and known limitations.
- `schema/capture_plan.schema.json` is generated: after changing `schema.py`, run
  `uv run floorfathom schema > schema/capture_plan.schema.json` (a test checks they match).
- `CozmoData/` and `Data/` are two different flats. `ground_truth.json` belongs to `Data/` only.

## Tiers and commands

Run from the repo root after `uv sync`. Each tier's owner keeps their own subsection: what it does, the command(s),
where the output lands, known limits. Outputs go to `out/` (git-ignored); clear them when done.

### LiDAR tier: floor plan, room plans and damage in one command

- `uv run floorfathom plan CozmoData/<scan> --out out/<name>` (`<scan>` is `single_room`, `single_scan_floor_only` or
  `single_scan_with_ceiling`). One command gives everything; no second command is needed.
- Time, cold (the LiDAR tier uses no caches, so cached is the same): `single_room` 2.0 min, `single_scan_floor_only`
  4.2 min, `single_scan_with_ceiling` 7.2 min. Per-stage seconds are in `plan.json` `diagnostics.notes`; damage is
  the largest stage (78, 169 and 319 s), then the bootstrap, then building the cloud.
- Outputs in `out/<name>/`: `plan.json` (CapturePlan, schema 0.4.0: per room walls, ceiling height, area, openings,
  `damage`, `concealed_flags`, `scope`, all with 95% intervals; plus `stitching`: adjacency, overlaps, footprint, drift
  on/off ablation), `rooms/<room id>.json` (that room's CapturePlan, as the photo tier writes), `plan.png` (top-down
  plan with dimensions, doorways and damage marked with red crosses), `debug/` (`points.ply`, `topdown.png`,
  `height_hist.png`, and `debug/damage/<surface>.png`: each wall, floor and ceiling unrolled, purple where not judged,
  damage outlined).
- Code: `io_lidar`, `points`, `planes`, `estimate`, `layout`, `drift` (pose graph from wall registration), `stitch`,
  `uncertainty`, `report`, `render`, `pipeline.run_lidar`, `lidar_frames` (RGB frames with depth for damage).
- Limits: no tape truth for `CozmoData/`, so accuracy is unmeasured; only about a quarter of walls meet the wall-length
  repeatability gate (README). Damage is unverified on real LiDAR damage (none exists in `CozmoData/`), so every region
  reported on those scans is a false positive (fixtures, glare, door edges).

### Shared damage stack (used by every tier)

- `assess.assess_room(room, floor_y, ceiling_y, frames, ...)` takes a room plan and posed RGB frames (`surfaces.Frame`),
  unrolls each surface (`surfaces.py`), finds damage (`damage.py`), fills `room.damage`, `room.concealed_flags`
  (`concealed.py`, named rules) and `room.scope` (`scope.py`, class-to-action table). Schema 0.4.0.
- `damage.detect(Patch)`: class-agnostic proposals (colour against the surface's local median, thin ridges, depth
  relief) classified by our own rules. A pretrained CLIP classifier was tried and dropped: it did worse than the rules
  on real patches. Rules, classes and scope actions are our own definitions, tuned on synthetic surfaces and a few
  real patches (`scripts/eval_damage_patches.py`).

### Video tier: floor plan and room plan commands

- Room plan (one clip = one room, with the yellow ruler taped up at the start, see `capture_protocol.md`):
  `uv run floorfathom plan Data/H1.MOV --out out/h1 --reference-length-cm 31.6`
- Floor plan (a folder of one-room clips, stitched into one plan): `uv run floorfathom plan out/clips --out out/video --reference-length-cm 31.6`
  after `mkdir out/clips && cp Data/B1.MOV Data/B2.MOV Data/H1.MOV out/clips` (`Data/` itself also holds `Full.MOV`, the whole
  flat, and `H2.MOV`, a second Hall take; both would be treated as extra rooms).
- Outputs, one command each: `plan.json` (rooms with walls, ceiling, area, openings, `damage`, `concealed_flags`, `scope`, all
  with 95% intervals; the stitched plan adds `stitching`, and a room that could not be placed stays in its own frame), `plan.png`,
  `rooms/<clip>.json` (that clip's own plan), `debug/damage/<surface>.png`, `work/` (keyframes, `sfm.pkl`, `dense.pkl`, `rays.pkl`,
  `depth924/`, `alignment.json`: caches only, a rerun into the same `--out` reuses them). Same layout as the photo tier, one clip or a folder.
- How: keyframes every 0.2 s, pycolmap SfM (CPU), monocular depth (Depth Anything V2 Metric-Indoor, `floorfathom fetch-models`)
  densified onto the SfM cloud, metric scale from the ruler's yellow body (`--reference-length-cm`, default 31.6; a depth-model
  scale with a 15% interval when it is not found), then the LiDAR tier's room estimator, bootstrap and stitching, with three
  video-only inputs: camera-ray free space (`video_rays`: cells a ray crossed count as open floor, so a glossy floor no longer
  fragments the room), wall lines fitted to the whole cloud (`video_walls`, reusing `photo_layout.wall_segments`) that cut off
  what lies behind a wall (a balcony seen through glass), and floor and ceiling re-centred on their point plateau
  (`video_heights`). Polygon tolerance is 25 cm for video (`VIDEO_PARAMS`), 12 cm for LiDAR.
- Code: `io_video`, `sfm`, `mono_depth`, `world`, `anchor` (ruler), `points_video`, `video_rays`, `video_walls`, `video_heights`,
  `video_openings`, `video_pipeline.run_video/run_clip`, `stitch`, `damage_video` (damage on the keyframes). Dev tool: `scripts/eval_video_estimator.py
  out/h1 hall_kitchen` compares estimator variants on a finished run's caches in about 40 s, no SfM.
- Time (this 7 GB CPU-only laptop, per clip of 13-35 s): SfM is most of it. Measured cold at 0.2 s spacing: H1 13.6 min, B1 14.6 min,
  B2 19.4 min (B2 while tests ran on the same machine). Earlier: 24-35 min at 0.15 s spacing, 11 min at 0.3 s (which lost the H1
  ceiling and moved its scale by 21%). All caches present: H1 98 s including the ray pass, bootstrap and damage; the H1 + B1 folder command 182 s (B1 stays in its own frame: no doorway fits).
  Keyframe spacing is `KEYFRAME_STEP_S` / `run_clip(keyframe_step_s=...)`.
- Results (Data/, tape in `ground_truth.json`; post the glossy-floor reflection fix, commit that added
  `video_heights.floor_from_strongest_spike`). Reflection fix: a glossy floor mirrors the room below itself and fooled
  `find_floor` into a floor 1.5 m too low; dropping points below the true floor (strongest spike) recovers it. On H1 this
  moved openings from 1 to 6 detected and `kitchen_side_wall` from +15.6% to +0.3% (see `Deliverables/fix_loop/`).
  - Walls +-3% gate: 3 of 16 across the four captures (H1 1/4 kitchen 527/526; H2 0/4; B1 0/4; B2 2/4 - 352/361, 293/301.5).
    Accuracy tracks capture coverage: B2 (small, fully swept) best; the hall (large, walked diagonally) 0-1/4. The three long
    hall walls are fragments because the camera never swept them face on. Depth-model walls also bow/kink by several cm.
  - Openings: H1 6/6 detected, 1/6 within the +-2 cm gate (main_door 108/107); the three interior doors read wide (129-159 vs
    71-81) on the sheared outline. A capture-coverage limit, not the floor bug.
  - Ceiling: observed on 2 of 4 (H2 288.7 +9.7, B2 263.8 -15.2), never within +-1.5 cm; reported not-observed on H1 and B1
    rather than guessed (bowed/smeared depth-model ceiling with competing spikes).
  - Areas: H1 23.2 (~24), H2 19.7, B1 5.30 (tape 6.12), B2 11.04 (tape 10.88, +1.5%).
  - Repeatability H1 vs H2 (same hall): 0/4 walls within 1 cm / 0.5% - FAIL. Root cause is different scale source (H1's ruler
    triangulation was implausible -> depth-model scale; H2's ruler accepted), so the two sit on different metric bases (area
    23.2 vs 19.7). The ruler is fragile on a small near-vertical object with a short baseline. Consistent ruler acceptance is
    the top repeatability fix.
  - Whole-property stitch (`out/clips` -> `out/video`, 213 s cached): H1 and B2 glue via the correct doorway (mutual, gap
    0.15 m; footprint 34.3 m2); B1 stays unplaced (no doorway seen). Correct adjacency vs tape (hall <-> b2_door <-> master bed).
    Drift: each clip is its own SfM frame placed by doorways, so no cross-room pose chain accumulates; within-clip drift is
    handled by SfM bundle adjustment (`stitching.drift` in `out/video/plan.json`).
  - Whole-flat single clip (`Data/Full.MOV`, 107 s, ~530 keyframes, 21.8 min): a single clip is one room by design, but
    SfM splits a whole-flat walk into several models (`sfm_split_into_several_models`), so the run recovers 2 partial rooms
    (area 24.9 and 6.5 m2; ceilings 291.5 and 305.6 cm - both observed) and stitches them (footprint 31.4 m2, 2 mutual
    doorway links, none unplaced). It does not recover all 4 rooms: the folder-of-clips command (one clip per room) is the
    supported multi-room path; Full.MOV degrades gracefully rather than failing.
  - Tests: full suite 229 passed, 15 skipped, 1 pre-existing LiDAR determinism flake (unrelated to video); all 43 video-tier
    tests pass. The result on H1 changes between runs (depth maps and keyframe spacing move it), and the damage found changes
    with it. Interval widths are large where the leave-chunks-out replicates disagree (H1 area 7-39 m2).

### Photo tier: floor plan and room plan commands

- Floor plan (every room folder, stitched): `uv run floorfathom plan Data --tier photo --out out/photo --reference-length-cm 31.6`
  (cold 4 min, cached 2.3 min, measured on 12 cores with nothing else running; two cold runs gave identical JSON;
  `--tier photo` is needed because `Data/` also holds `.MOV` files).
- Room plan (one room folder of 2-8 stills): `uv run floorfathom plan Data/Hall --tier photo --out out/hall --reference-length-cm 31.6`
  (the folder may be `Data/B1`, `Data/B2` or `Data/Hall`).
- Outputs: `plan.json` (stitched property plan; each room keeps its own frame if it could not be placed), `rooms/<room>.json`
  (that room's own plan), `plan.png`, `debug/damage/<surface>.png`, `work/depth/` (cached depth maps: a rerun of the same
  photos into the same `--out` skips the model and gives the same numbers).
- How: per-image pycolmap SfM (one camera per still, EXIF focal, `PER_IMAGE` mode; `photo_pose.run_photo_sfm`) recovers real
  camera poses when the stills overlap with a translation baseline. When SfM registers fewer than 60% of the stills (or fails),
  it falls back to the old SIFT rotation-only single-station registration (`register_rotations`), flagged
  `sfm_underregistered_..._used_rotation_only` / `sfm_fell_back_to_rotation_only`. Either way: monocular metric depth
  (Depth Anything V2 Metric-Indoor Small, `floorfathom fetch-models`, offline afterwards) densified onto the cloud
  (`points_video`), gravity (`world`/wall thinness), wall/floor/ceiling planes by RANSAC, an evidence-weighted Manhattan angle
  snap (`photo_layout.snap_manhattan`), walls outlined by ray casting. Scale comes from the yellow ruler triangulated across
  ≥3 registered stills (`photo_reference.ruler_scale_sfm`/`anchor`), else from the model with a 30% scale uncertainty.
  Intervals: leave-one-photo-out plus assumed systematic terms plus scale uncertainty. Thin input gives wider intervals or
  null values, never a confident guess.
- Code: `io_photos`, `photo_pose` (SfM + rotation-only fallback), `photo_scene` (`build_scene_sfm` and rotation `build_scene`),
  `photo_layout` (+ `snap_manhattan`), `photo_reference` (`ruler_scale_sfm`), `photo_pipeline.run_photo`, `stitch.refine_global`,
  `photo_damage` (shared damage stack on the photo frames).
- Limits (Data/, tape in `ground_truth.json`, real run of all four photo rooms, 2026-09-21): the ±8% wall gate is met by
  **0 of 4 rooms** and floor area is **null in every room** — the real captures are 6-9 wide-baseline stills that
  under-register in SfM (Hall 4/8, Hall2 4/9, B1 SfM-failed → all fall back to the single-station path; B2 registered but its
  full-length mirror doubled the room), so no outline closes (`open_boundary`) and walls come out as fragments. Ceilings are
  biased high: Hall 3.04 m (ruler), Hall2 3.97 m, B1 3.81 m vs 2.79 m tape — every interval contains the tape, the no-ruler
  rooms very widely; B2's ceiling is not observed. Only the Hall found a usable ruler; Hall2/B1/B2 fell back to the depth-model
  scale and B2 is mis-scaled 2-3× (walls 6-9 m in a 3.6 m room). Repeatability Hall vs Hall2 FAILS (ceiling 30% apart,
  different scale source: ruler vs depth model). Whole-property stitch fails: no room detected a doorway, so nothing glues.
  The blocker is capture coverage/overlap (`capture_protocol.md`), not the estimator. Only four rooms exist, so intervals are
  uncalibrated. Per-room cold time: Hall 73 s, Hall2 65 s, B1 83 s, B2 142 s.

### Video-tier damage

- Part of the video command, no separate step: `uv run floorfathom plan Data/H1.MOV --out out/h1_damage --reference-length-cm 31.6`
  (the video tier's command; one clip is one room). Timings are in the video tier section: the damage step adds no depth-model
  time of its own, because it may only use keyframes the dense pass already depth-mapped (`work/depth924/`, shared cache).
- Outputs: `plan.json` (per room `damage`, `concealed_flags`, `scope`, sizes in metres with 95% intervals that include the
  metric scale's error) and `debug/damage/<surface>.png` (each judged wall and the ceiling unrolled from the keyframes, purple
  where not judged, damage outlined). The notes in `diagnostics` say what could not be judged.
- Code: `damage_video.py` moves SfM poses into the plan frame, picks about 10 keyframes per wall or ceiling, uses the depth
  model only to leave out what stands in front of a wall (its scale is aligned to each wall first; never for relief) and calls
  `assess.assess_room` (called from `video_pipeline.run_clip` before stitching, with `frames=set(dense.frame_ratio)`);
  `cached_depth` keeps depth maps on disk (float16, md5 of the frame, same numbers cold and cached); `renumber_damage` keeps ids
  unique when clips become rooms of one property. Not searched: the floor.
- Dev tools (they read a finished run's folder, so they are not the user command): `scripts/damage_video_clip.py`,
  `scripts/damage_video_wall.py` (fits one wall from the dense cloud), `scripts/eval_damage_video.py` (synthetic sweep),
  `scripts/eval_damage_patches.py` (acceptance test on saved real patches in `out/h1_patches`, made with
  `damage_video_clip.py --save-patches`).
- Limits: synthetic walls give exact stain sizes and cracks of 3 mm or more; 1-2 mm cracks are missed (the staged 12 x 22 cm
  hairline crack in H1 was not found). On real H1 the staged 21 x 25 cm leakage mark was found (22 x 16 cm) on a wall fitted by
  hand, but the plan's own walls are fragments (5 of 19 supported), so walls the room estimator misses are not judged at all.
  False positives remain on wood grain, door hardware and cabinet edges (a bedroom with no damage still gives a few regions).

### Photo-tier damage

- Part of the photo command, no separate step: `uv run floorfathom plan Data --tier photo --out out/photo --reference-length-cm 31.6`
  (the room command on one folder works too). Damage adds about 23 s to a run with cached depth (B1 8.5 s, B2 3.9 s, Hall 10.8 s);
  the depth model dominates a cold run.
- Outputs: `plan.json` (per room `damage`, `concealed_flags`, `scope`; sizes in metres with 95% intervals that include the
  scale uncertainty, about +-30% when no ruler was found) and `debug/damage/<surface>.png` (each judged wall and the ceiling
  unrolled, purple where not judged, damage outlined). `damage_coverage_partial` and the `diagnostics` notes say what could
  not be judged.
- Code: `photo_damage.assess` runs the shared `assess.assess_room` on the room's frames from one spot (one view is enough,
  no relief, depth only rejects pixels that are not on the wall; walls and ceiling only, the floor is glossy tile), then gives the
  room's copy in the stitched plan the same regions (`damage_frame.reframe_damage` moves ceiling regions with the room) and
  recomputes its flags and scope with the neighbouring rooms in view.
- Dev tools: `scripts/eval_damage_photo.py out/photo/plan.json` (scores the tape-measured items of `ground_truth.json`),
  `scripts/export_photo_patches.py Data out/damage_patches/photo --cache-dir out/photo` (saves the real unrolled patches
  for `scripts/eval_damage_patches.py --dir`; needs the depth cache of a finished run).
- Limits (real 4-room run 2026-09-21, `out/photo`, rules-based detector, `scripts/eval_damage_photo.py`): 1 of 2 staged
  items found. The 21 x 25 cm leakage mark in the Hall was found at 17 x 26 cm (class `water_stain` correct; height inside
  the interval [22.0, 30.1], width 17 vs 21 short/outside). The 12 x 22 cm kitchen crack was not found: the Hall walls facing
  the kitchen are mostly not seen face on, and a hairline crack is not resolvable in stills. The run gave 10 other regions,
  all false: on B1 (undamaged) 3 wood-grain cracks and 2 wardrobe/wall soot regions, on the Hall 2 extra water stains, on
  Hall2 3 grain cracks. Only one room in a run has a tape-measured mark, so the size error is one sample.
