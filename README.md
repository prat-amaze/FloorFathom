# FloorFathom

Handheld iPhone captures to a dimensioned floor plan, one command per capture. Built for
the Cozmo AI case study (`Applied AI.pdf`).

## Status

| Tier | State |
|---|---|
| LiDAR (depth + poses + intrinsics) | **Per-room plans working** (this document). |
| Video | **Runs end to end, one room per clip; misses the wall and opening gates on real clips** (see "Video tier"). |
| Photos | Not started. |

Whole-property stitching (`stitch.py`) is built and tested on synthetic layouts: doorway
adjacency, overlap check, footprint with an interval, and placement of rooms that arrive
in their own frames (video, photo) by gluing their doorways. It has not been run on real
video or photo rooms yet. Not done for the LiDAR tier: damage regions, concealed-damage
flags, scope items.

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
7. `stitch.py`: each doorway is linked to the nearest other room outline within 1.5 m (the
   gap is kept as evidence), overlapping rooms are found, the footprint is the union of
   the room polygons. Rooms in their own frames are placed by gluing doorway to doorway
   (inward normals opposite, centres one assumed 0.15 m wall apart), most certain room
   first; a room with no doorway that fits is reported as unplaced, near-ties are flagged.
8. `drift.py`: per-chunk (x, z, yaw) drift from wall registration between time chunks of
   the walk, weighted by how well each pair constrains it. Applied to every LiDAR run
   before the room estimate, with the on/off footprint ablation reported in `plan.json`
   (`stitching.drift`). `run_lidar(correct_drift=False)` switches the correction off and
   `drift_ablation=False` skips the extra uncorrected run (there is no CLI flag for either).

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
  walks did not see the same things), so this comparison does not isolate the estimator.
- Same walk, two frame subsets, `scripts/same_walk_repeatability.py` (coverage identical, so
  only the estimator differs; `single_scan_with_ceiling`, six rooms match one to one). Room
  areas agree within 1% for four rooms and 5% for the two smallest. Wall **lines** repeat well:
  median position difference 0.3 cm, 76% within 1 cm. Wall **corners** (both neighbours
  supported) move by a median 1.5 cm (46% within 1 cm, 64% within 3 cm). A wall end next to an
  unsupported edge (a doorway or an unseen stretch) moves by a median 31 cm, and only 26% of
  the walls that lie on the same line in both runs meet the 1 cm / 0.5% gate (median length
  difference 9 cm, n = 19). So the wall length gate fails on segmentation and unsupported
  ends, not on wall position. With the drift correction on, the same test gives a line position
  difference of 1.8 cm and corners of 3.2 cm (median): each run estimates the drift separately and the
  two estimates differ by a median 1.5 cm (up to 3.5 cm and 0.9 degrees) on drifts of 16 cm and 4
  degrees, which the correction removes. Snapping wall directions to the room's dominant
  direction was tried and moved corners only from 1.7 to 1.3 cm median, so it was not kept.
  This measures the estimator, not the device: it is not a repeat capture and no tape truth
  is involved.

### Drift ablation (footprint with the correction on and off)

The walk is drift-corrected by default and every run reports its own on/off footprint. `drift.py`
recovers injected drift on synthetic walks (14 cm and 0.7 degrees down to under 2 cm,
`tests/test_drift.py`). On the sample scans it finds chunks up to 16-22 cm and 2-4 degrees apart,
and the registration disagreement (chi2) falls by 95%. The footprint moves little:

| Scan | Footprint, drift off | on, prior 5 cm / 0.5 deg (default) | on, prior 2 cm / 0.2 deg |
|---|---|---|---|
| `single_scan_with_ceiling` | 58.4 m2, 6 rooms | 58.0 m2, 6 rooms (-0.7%) | 56.6 m2, 6 rooms (-3.0%) |
| `single_scan_floor_only` | 52.3 m2, 6 rooms | 51.3 m2, 5 rooms (-1.9%) | 50.7 m2, 5 rooms (-3.0%) |

Honest caveat: there is no tape truth for these scans, so this does not show that the
correction is more accurate. Two different recordings of the same flat (`single_scan_with_ceiling`
and `single_scan_floor_only`, `scripts/cross_scan_repeatability.py [--drift]`), off against on with
prior 2 cm / 0.2 deg: 6 vs 5 matched rooms, mean IoU 0.77 vs 0.78, summed area difference 7.2 vs
10.9 m2, median wall difference 44 vs 55 cm. With prior 5 cm one room grew by 56%. The recordings
do not see the same things, so this is a weak test, but the correction did not improve it. It is on
because the poses must not be taken as recorded when the walk is long, and the effect on the
footprint is small either way; the step tolerances of the estimate are assumptions, not calibrated,
and the default prior was fixed before this comparison, not tuned on it. A floor-only scan lost a
room (6 to 5) with the correction on, which is the failure to watch.

## Known limitations

- Windows are not detected at the LiDAR tier (glass, partial-height gaps); only
  floor-level doorway gaps are. A closed door leaf hides its doorway.
- Wall length is corner to corner on the fitted lines; a wardrobe against a wall is
  measured to its face.
- Rooms are outlines of visited free space. Spaces that open onto the room through a gap
  wider than 1.1 m (open plan, or a real doorway that wide) are merged with it.
- The systematic terms of the intervals (pose drift, plane offset) are assumptions in
  `uncertainty.py`, to be calibrated against tape measurements.
- Drift is corrected in blocks of walk chunks, which can leave small steps at chunk boundaries; a smooth
  pose-level correction was not finished. Whether the correction improves accuracy is untested without tape truth.
- Rooms placed by doorways assume a 0.15 m wall between them; a wrong thickness shifts each
  hung room by the difference. Two doorways of equal width whose rooms fit either way are
  flagged `placement_ambiguous`. Rooms with no visible doorway (a closed door leaf, or a few
  photos) cannot be placed and are listed in `stitching.unplaced`.

## Video tier

```
uv run floorfathom plan Data/B2.MOV --out out/b2 --reference-length-cm 31.6
```

One clip is one room. `video_pipeline.py`: keyframes (`io_video.py`), poses and sparse points by
SfM (`sfm.py`, pycolmap), a dense cloud from a depth model fitted to the SfM depths
(`points_video.py`), gravity from the floor and ceiling planes (`world.py`), then the same estimator
and leave-chunks-out intervals as the LiDAR tier. SfM has no scale. The scale comes from the
protocol's yellow ruler (`anchor.py`): its two ends are triangulated in the first 15 s of the clip
and its tape-measured length (`--reference-length-cm`, default 31.6, ours) gives metres per SfM unit
with its own uncertainty. If the ruler is not found, is flagged (few frames, weak sideways
movement, large residual) or is far from the depth model's scale, the depth model's scale is used,
every interval carries at least 15% scale uncertainty and the room is flagged
`scale_from_depth_model_only` with the reason.

Numbers on the new ruler clips (`B2`, `B1`, `H1`, this laptop, tape values in `ground_truth.json`,
scored with `scripts/eval_video.py`):

| | `B2` | `B1` | `H1` (Hall) |
|---|---|---|---|
| keyframes with a pose | 100% | 100% | 100% |
| scale method | ruler, +-0.6% | ruler, +-0.4% | ruler |
| walls within 3% of the tape | 0/4 | 0/4 | 0/4 |
| ceiling | not observed | not observed | 2.74 m [2.60, 2.88] against 2.79 (-1.9%) |
| openings within 2 cm | 0/1 (found 91 cm, tape 81) | 0/1 (missed) | 0/4 (all missed) |
| run time | 35 min | 24 min | 35 min |

What this shows and does not show:

- The ruler scale is 0.62-0.65 of the depth model's on these clips. The only independent check is
  the `H1` ceiling, which the scale estimate never sees: 2.74 m against 2.79 m by tape. The depth
  model's scale would have given about 4.5 m, and the old `H2` clip (depth scale only) gave 4.68 m.
  One clip is one data point; `B2` and `B1` had no ceiling to check.
- The wall and opening gates are not met. Walls come out as fragments (7 to 19 per room) and short
  (`B2` 2.47 m of 3.61 m, `H1` 1.72 m of 5.26 m); the estimator was tuned on LiDAR clouds, and this
  was not fixed.
- Repeatability of `H1` against `H2` was not scored: `H2` was not run on the ruler clips.
- Without a ruler in view the depth model gave scale errors of -1%, +7% and +67% on the earlier
  clips, so the fallback cannot meet the +-3% gate.
- One clip is one room. A folder of clips gives one stitched plan (`stitch.py` glues the rooms by their
  doorways; a clip with no room is listed in the notes, an unplaceable room stays in its own frame).
  Tested on canned per-room plans only. On the real plans of `H1`, `B1` and `B2`, only `B2` was placed:
  it was the only room with a detected doorway, so `H1` and `B1` had nothing to glue and were reported
  unplaced. Doorway detection is what limits video stitching.

## Layout

`src/floorfathom/` the package, `tests/` synthetic-capture tests, `scripts/` analyses,
`schema/` the published JSON Schema, `capture_protocol.md` and `edge_cases.md` the
capture docs, `ground_truth.json` tape measurements of the flat captured in `Data/`.
