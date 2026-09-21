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
- Limits (Data/, tape in `ground_truth.json`; accuracy is the open part, the pipeline itself runs end to end): walls still miss the
  +-3% gate (0 of 4 on H1: one wall -6%, the other three come out as fragments (now subject to global wall-line augmentation and re-merge; re-evaluation needed); the depth-model walls bow and kink by several cm),
  openings 0 of 4 on H1 (widths 55-125 cm; a through-ray detector finds doors at 63-75 cm against 71-81; now integrated (re-evaluation needed)),
  ceiling on H1 285.5 cm against 279 (+6.5 cm, tape inside the interval; B1 and B2 show none). With ray free space the room
  polygon is right in kind: H1 22.0 m2 in 13 edges (was 19.3 m2 in 38, a glossy-floor cloud fragments), B1 5.66 m2 in 4 edges (tape 6.12).
  H1 vs H2 repeatability and `Full.MOV` not run. The result on H1 changes between runs (depth maps and keyframe spacing move it),
  and the damage found changes with it. Interval widths are large where the leave-chunks-out replicates disagree (H1 area 11-33 m2).

### Photo tier: floor plan and room plan commands

- Floor plan (every room folder, stitched): `uv run floorfathom plan Data --tier photo --out out/photo --reference-length-cm 31.6`
  (cold 4 min, cached 2.3 min, measured on 12 cores with nothing else running; two cold runs gave identical JSON;
  `--tier photo` is needed because `Data/` also holds `.MOV` files).
- Room plan (one room folder of 2-8 stills): `uv run floorfathom plan Data/Hall --tier photo --out out/hall --reference-length-cm 31.6`
  (the folder may be `Data/B1`, `Data/B2` or `Data/Hall`).
- Outputs: `plan.json` (stitched property plan; each room keeps its own frame if it could not be placed), `rooms/<room>.json`
  (that room's own plan), `plan.png`, `debug/damage/<surface>.png`, `work/depth/` (cached depth maps: a rerun of the same
  photos into the same `--out` skips the model and gives the same numbers).
- How: SIFT rotation-only registration of the stills from one spot, monocular metric depth (Depth Anything V2 Metric-Indoor
  Small, fetched by `floorfathom fetch-models`, offline afterwards), depth scales harmonised across overlaps, gravity from wall
  thinness, wall/floor/ceiling planes by RANSAC, walls outlined by ray casting. Scale comes from the yellow reference ruler
  when it is found in the photos (`photo_reference`), else from the model with a 30% scale uncertainty. Intervals: leave-one-photo-out
  plus assumed systematic terms plus scale uncertainty. Thin input gives wider intervals or null values, never a confident guess.
- Code: `io_photos`, `photo_pose`, `photo_scene`, `photo_layout`, `photo_reference`, `photo_pipeline.run_photo`,
  `photo_damage` (shared damage stack on the photo frames).
- Limits (Data/, tape in `ground_truth.json`, first real run): the ±8% wall gate is not met. Areas are null in all three rooms
  (outline incomplete: some walls are never seen from one spot), and unseen walls get no length. Ceiling with the ruler: Hall
  3.10 m and B2 2.44 m vs 2.79 m tape, intervals contain the tape; B1 has no ruler found (3.32 m, interval 1.3-5.3 m).
  Unregistered photos: 3 of 6 in B1, 2 of 9 in Hall. Depth scale differs from photo to photo, so B2 and Hall carry
  `depth_scale_inconsistent`/`pose_loop_inconsistent`. Only four rooms exist, so the intervals are uncalibrated.

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
- Limits (first real run, rules-based detector): the staged 21 x 25 cm leakage mark in the Hall was found at 22.5 x 25.5 cm,
  both tape values inside the intervals. The 12 x 22 cm kitchen crack was not found: the Hall walls facing the kitchen are
  mostly not seen face on, and a hairline crack is not resolvable in stills. The run gave 7 other regions, all false: door
  frame edges and grain as cracks, a power socket as mould, window bars, a curtain and a wardrobe edge as cracks or soot. On
  the 22 real patches 6 of 23 checks fail. Only one room in a run has a tape-measured mark, so the size error is one sample.
