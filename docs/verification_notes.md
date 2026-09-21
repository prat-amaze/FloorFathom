# Verification notes

Measured results that used to sit in `README.md`, moved here word for word so the README can stay a
"get running" document. Nothing below was re-measured when it was moved (2026-09-21), and the reproduction bundle
(sent separately, not in git) is what regenerates them.

Known disagreements to read before quoting a number from this file:

- The video results table further down (H1 ceiling 2.74 m against 2.79) predates the ray free-space, wall-line and
  height-refinement changes. The same section, and `CLAUDE.md`, later give H1 ceiling 285.5 cm against 279.
- `Deliverables/benchmark_report.md` scores H1 vs H2 repeatability and gives H1 walls 1/4; this file and `CLAUDE.md` say
  "not scored" and 0/4.
- The damage-acceptance failure counts differ: "7 against 5" below, "6 of 23 checks on 22 real patches" in `CLAUDE.md`.

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

## Damage: measured results

- Checked on synthetic painted rooms (known place and size, each class, shadows, furniture, a door beside a stain, tilted
  planes) and end to end on a synthetic LiDAR scan. Not checked on real LiDAR damage: none exists in `CozmoData/`.
- On real patches from the video tier (`scripts/eval_damage_patches.py`) the staged leakage mark is found with the right
  class (22 x 16 cm against a tape 21 x 25 cm), but doors, hardware, sockets, window bars and glare still give false
  regions. On `CozmoData/single_room` (no known damage) an earlier detector reported 47 false regions; it was then fixed
  for tilted relief, dark fixtures, glare and door edges and not re-measured.
- A pretrained CLIP classifier over the same proposals was tried and dropped: on the real patches it failed 7 of the
  acceptance checks against 5 for the rules.

## Video tier: measured results

Numbers on the new ruler clips (`B2`, `B1`, `H1`, this laptop, tape values in `ground_truth.json`,
scored with `scripts/eval_video.py`):

| | `B2` | `B1` | `H1` (Hall) |
|---|---|---|---|
| keyframes with a pose | 100% | 100% | 100% |
| scale method | ruler, +-0.6% | ruler, +-0.4% | ruler |
| walls within 3% of the tape | 0/4 | 0/4 | 0/4 |
| ceiling | not observed | not observed | 2.74 m [2.60, 2.88] against 2.79 (-1.9%) |
| openings within 2 cm | 0/1 (found 91 cm, tape 81) | 0/1 (missed) | 0/4 (all missed) |
| run time, cold (0.2 s keyframes) | 19 min (tests ran alongside) | 14.6 min | 13.6 min |

What this shows and does not show:

- The ruler scale is 0.62-0.65 of the depth model's on these clips. The only independent check is
  the `H1` ceiling, which the scale estimate never sees: 2.74 m against 2.79 m by tape. The depth
  model's scale would have given about 4.5 m, and the old `H2` clip (depth scale only) gave 4.68 m.
  One clip is one data point; `B2` and `B1` had no ceiling to check.
- The wall and opening gates are still not met, but the room estimator no longer breaks on the glossy floor. Three
  video-only inputs were added to the shared estimator, each off by default so the LiDAR and photo tiers are unchanged:
  camera-ray free space (`video_rays.py`), wall lines fitted to the whole cloud that cut off what lies behind a wall
  (`video_walls.py`) and floor and ceiling re-centred on their point plateau (`video_heights.py`). On `H1`, from the same
  cached SfM and depth (`scripts/eval_video_estimator.py`), the room went from 19.3 m2 in 38 edges to 22.0 m2 in 13 edges,
  the balcony seen through glass is cut off, the ceiling error went from +11.9 cm to +6.5 cm (tape inside the interval),
  and one wall reached -6%; the other three walls are still fragments and the openings are 0 of 4. On `B1` the same
  change gave 5.66 m2 in 4 edges (tape 6.12). Depth-model walls bow and kink by several centimetres, which no polygon
  step removes; the +-3% gate needs a better geometry, not a better outline.
- Repeatability of `H1` against `H2` was not scored: `H2` was not run on the ruler clips.
- Without a ruler in view the depth model gave scale errors of -1%, +7% and +67% on the earlier
  clips, so the fallback cannot meet the +-3% gate.
- One clip is one room. A folder of clips gives one stitched plan (`stitch.py` glues the rooms by their
  doorways; a clip with no room is listed in the notes, an unplaceable room stays in its own frame).
  Tested on canned per-room plans only. On the real plans of `H1`, `B1` and `B2`, only `B2` was placed:
  it was the only room with a detected doorway, so `H1` and `B1` had nothing to glue and were reported
  unplaced. Doorway detection is what limits video stitching.
