# FloorFathom

Handheld iPhone captures to a dimensioned floor plan, one command per capture. Built for
the Cozmo AI case study (`Applied AI.pdf`).

## Status

| Tier | State |
|---|---|
| LiDAR (depth + poses + intrinsics) | **Per-room plans working** (this document). |
| Video | Not started. |
| Photos | Not started. |

Not done for the LiDAR tier either: stitching rooms into one property plan with drift
correction, damage regions, concealed-damage flags, scope items. The rooms of a
multi-room scan are already placed in one coordinate frame, but not yet checked for
drift or connected into an adjacency graph.

## Run it

```
uv sync
uv run floorfathom plan <capture_folder> --out out/<name>
```

`<capture_folder>` holds `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`,
`rgb.mp4` (the layout of `CozmoData/*`). The tier is detected from the folder.

Outputs in `--out`:

- `plan.json`: the output contract. Every measurement is `{value, lo, hi, unit, method}`,
  a 95% interval. A quantity that was not measured is `null` with a note, never a guess.
  The schema is published in `schema/capture_plan.schema.json` (`floorfathom schema`).
- `plan.png`: the rendered plan, top-down, with wall lengths, area, ceiling height and
  doorways.
- `debug/points.ply`, `debug/topdown.png`, `debug/height_hist.png`: the raw geometry next
  to what was extracted, for checking by eye.

Options: `--bootstrap N` (replicates for the intervals, default 20), `--seed`, `--no-debug`.
Runs are deterministic: same input and seed give the same JSON.

Timing on the sample scans (laptop, CPU): `single_room` 37 s, `single_scan_floor_only`
86-93 s, `single_scan_with_ceiling` 120-130 s, of which about half is building the cloud.

## How it works

1. `io_lidar.py`: read depth (mm), confidence, poses, intrinsics rescaled to the 256x192 depth grid.
2. `points.py`: back-project to a gravity-aligned world point cloud. Conventions were found
   by search, not assumed: depth as-is, OpenCV camera axes, quaternion is camera-to-world,
   world +y up. The picture in `rgb.mp4` looks rotated 90 degrees; that is only the sensor's
   fixed landscape orientation and does not affect geometry.
3. `planes.py`: floor and ceiling are spikes in the height histogram; ceiling height is
   their distance. No ceiling spike means `null`, not an estimate.
4. `layout.py`, `estimate.py`: walls are cells occupied over most of 0.25-2.05 m of height
   (furniture is not). Free space is observed floor minus walls. Doorways are closed by
   extending wall segments across gaps up to 1.1 m, rooms are connected free space the
   camera visited, each outline is simplified and snapped to fitted wall lines. Right
   angles are never assumed. Doorways are gaps in the wall evidence a closure crosses.
5. `uncertainty.py`: intervals from a delete-2-of-20-chunks jackknife with a robust
   spread, plus an assumed systematic term. Rooms whose shape flips between replicates
   are flagged and their interval is stretched to cover the disagreement.
6. `report.py`, `schema.py`, `render.py`, `debug.py`, `cli.py`: contract, drawing, CLI.

## Verification

`uv run pytest` (about 6 minutes): 12 tests on synthetic captures with exactly known
geometry, ray-cast into depth frames with noise and written in the real file layout.

| Case | Truth | Recovered |
|---|---|---|
| Rectangle 5.0 x 4.0 m, ceiling 2.6 m | 20.00 m2, walls 5/4/5/4 | 19.998 m2, walls within 3 cm, ceiling within 1.5 cm |
| Same room rotated 27 degrees | 20.00 m2 | within 1.5% |
| L-shaped room | 18.00 m2, 6 walls | within 3%, 6 walls |
| Two rooms, 0.90 m doorway | 12 and 16 m2, door 0.90 m | 12.0 and 16.0 m2, door 0.88-0.89 m |
| Furniture in the room | 20.00 m2 | within 3% |
| Missing ceiling | ceiling unknown | `null`, flagged |
| Two noisy captures of one room | agree within 1 cm or 0.5% per wall | passes |
| Intervals | contain the true walls, area, ceiling | pass, and are not degenerate |

Real data has no ground truth for the Cozmo samples, so the evidence there is
consistency only.

- Ceiling: three rooms of `single_scan_with_ceiling` come out at 3.076, 3.075 and 3.063 m,
  three lower rooms at 2.42, 2.35 and 2.28 m, each with an interval of about +/-2 cm.
- Repeatability across two walks of the same flat, `scripts/cross_scan_repeatability.py`:
  the wall maps align (75% overlap) and every one of six rooms matches exactly one room in
  the other capture (overlap 0.71-0.90). The largest room agrees to 0.3% in area. **Other
  rooms differ by 4-27% in area, and wall lengths differ by a median of 44 cm, so the
  1 cm / 0.5% repeatability gate is not met on real scans.** Part of that is real (the
  walks did not see the same things), but the estimator also leaks into narrow spaces
  behind doorways and splits walls differently between scans.

## Known limitations

- Windows are not detected at the LiDAR tier (glass, partial-height gaps); only
  floor-level doorway gaps are. A closed door leaf hides its doorway.
- Wall length is corner to corner on the fitted lines; a wardrobe against a wall is
  measured to its face.
- Rooms are outlines of visited free space. Spaces that open onto the room through a gap
  wider than 1.1 m (open plan, or a real doorway that wide) are merged with it.
- The systematic terms of the intervals (pose drift, plane offset) are assumptions in
  `uncertainty.py`, to be calibrated against tape measurements.
- Poses are used as-is. Drift is not corrected yet; it matters for stitching rooms.

## Layout

`src/floorfathom/` the package, `tests/` synthetic-capture tests, `scripts/` analyses,
`schema/` the published JSON Schema, `capture_protocol.md` and `edge_cases.md` the
capture docs, `ground_truth.json` tape measurements of the flat captured in `Data/`.
