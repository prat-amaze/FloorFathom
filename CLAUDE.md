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
  `single_scan_with_ceiling`; 2-5 minutes each).
- Outputs: `plan.json` (per room: walls, ceiling height, area, openings, `damage`, `concealed_flags`, `scope`, all with
  95% intervals; plus `stitching`: adjacency, overlaps, footprint, and the drift on/off ablation), `plan.png` (top-down
  plan), `debug/` (`points.ply`, top-down and height views, and `debug/damage/<surface>.png`: every wall, floor and
  ceiling unrolled, purple where not judged, damage outlined).
- Code: `io_lidar`, `points`, `planes`, `estimate`, `layout`, `drift` (pose graph from wall registration), `stitch`,
  `uncertainty`, `report`, `render`, `pipeline.run_lidar`, `lidar_frames` (RGB frames with depth for damage).
- Limits: no tape truth for `CozmoData/`, so accuracy is unmeasured; only about a quarter of walls meet the wall-length
  repeatability gate (README). Damage is unverified on real LiDAR damage (none exists) and gives false positives on
  fixtures and glare.

### Shared damage stack (used by every tier)

- `assess.assess_room(room, floor_y, ceiling_y, frames, ...)` takes a room plan and posed RGB frames (`surfaces.Frame`),
  unrolls each surface (`surfaces.py`), finds damage (`damage.py`), fills `room.damage`, `room.concealed_flags`
  (`concealed.py`, named rules) and `room.scope` (`scope.py`, class-to-action table). Schema 0.4.0.
- `damage.detect(Patch)`: class-agnostic proposals (colour against the surface's local median, thin ridges, depth
  relief) classified by our own rules. A pretrained CLIP classifier was tried and dropped: it did worse than the rules
  on real patches. Rules, classes and scope actions are our own definitions, tuned on synthetic surfaces and a few
  real patches (`scripts/eval_damage_patches.py`).

### Video tier: floor plan and room plan commands

(owner fills)

### Photo tier: floor plan and room plan commands

(owner fills)

### Video-tier damage

- Part of the video command, no separate step: `uv run floorfathom plan Data/H1.MOV --out out/h1_damage --reference-length-cm 31.6`
  (one clip is one room; SfM and the depth pass take 10+ minutes, cached in `out/<name>/work/`).
- Outputs: `plan.json` (per room `damage`, `concealed_flags`, `scope`, sizes in metres with 95% intervals that include the
  metric scale's error) and `debug/damage/<surface>.png` (each judged wall and the ceiling unrolled from the keyframes, purple
  where not judged, damage outlined). The notes in `diagnostics` say what could not be judged.
- Code: `damage_video.py` moves SfM poses into the plan frame, picks about 10 keyframes per wall or ceiling, uses the depth
  model only to leave out what stands in front of a wall (its scale is aligned to each wall first; never for relief) and calls
  `assess.assess_room`; `renumber_damage` keeps ids unique when clips become rooms of one property. Not searched: the floor.
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
