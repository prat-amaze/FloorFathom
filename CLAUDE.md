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

(owner fills)

### Photo-tier damage

(owner fills)
