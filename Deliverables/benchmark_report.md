# Benchmark Report

Gate numbers, repeatability table, timing. Video tier numbers are the **post-fix** runs
(commit `7055a33`, glossy-floor reflection fix — see `fix_loop/declaration.md`). All video
runs used `--reference-length-cm 31.6`.

**Scope, and what was not done.** Every tape-scored number below is photo or video captured with an
iPhone 16 (no LiDAR scanner) on one flat (`Data/`). No iPhone Pro (LiDAR) device was available to us, so
the LiDAR tier was only run on the Cozmo-provided sample scans (`CozmoData/`: a different flat, no tape
truth). Two required comparisons could therefore not be done as specified: the head-to-head against a
consumer app and the LiDAR repeatability gate. Both are stated in "Not done: needs an iPhone Pro" at
the end of this report, with what we measured instead.

---

## Opening widths (≤2 cm on ≥85% of openings; missed/phantom = miss)

Video tier, `hall_kitchen` (H1), post-fix. Assignment minimises total error to the 6 tape openings.

| Tier | Opening | Tape (cm) | Measured (cm) | Error (cm) | Pass? |
|---|---|---|---|---|---|
| Video | main_door | 107 | 108 | +1 | PASS |
| Video | balcony_door | 178 | 183 | +5 | fail |
| Video | kitchen_window | 110 | 129 | +19 | fail |
| Video | b2_door | 81 | 105 | +24 | fail |
| Video | b1_door | 81 | 141 | +60 | fail |
| Video | bathroom_door | 71 | 159 | +88 | fail |
| **Video** | **all** | — | — | — | **1/6 within 2 cm (was 1/6 detected pre-fix; fix found all 6, positions still short — see fix loop)** |
| LiDAR | — | — | — | — | NOT RUN (no tape truth on CozmoData; no Pro device to scan a tape-measured room) |
| Photo | — | — | — | — | see photo tier report |

Detection: **6/6 openings found** post-fix vs **1/6 pre-fix**. Positions are biased wide because the
three interior doors sit on a sheared room outline (unswept left wall); see the fix loop post-mortem.

---

## Ceiling height (≤1.5 cm per room; spread ≤1 cm across captures)

| Tier | Room | Tape (cm) | Measured (cm) | Error (cm) | Pass? |
|---|---|---|---|---|---|
| LiDAR | single_scan_with_ceiling (6 rooms) | no tape truth | 307.6, 307.5, 241.8, 306.4, 234.6, 227.9 (each ±1.7–1.8 cm) | — | not scorable: no tape truth on CozmoData; the other two scans observe no ceiling |
| Video | hall_kitchen (H1) | 279 | not observed | — | fail (ceiling smeared/bowed; not reported rather than guessed) |
| Video | hall_kitchen (H2) | 279 | PENDING | — | — |
| Video | bedroom_b1 | 279 | PENDING | — | — |
| Video | master_bedroom_b2 | 279 | PENDING | — | — |

The video ceiling on H1 is reported as not-observed rather than a wrong number: the depth-model ceiling
is bowed and smeared 1.9–3.4 m above the floor with two competing spikes (true 2.85 m, spurious 3.16 m),
and the room mask it would be validated against is mis-shaped. This is a known open item, distinct from
the shipped reflection fix.

---

## Wall lengths

### Video tier (gate: ±3%), hall_kitchen (H1), post-fix

| Room | Wall | Tape (cm) | Measured (cm) | Error (%) | Pass? |
|---|---|---|---|---|---|
| hall_kitchen | kitchen_side_wall | 526 | 527 | +0.3 | **PASS** |
| hall_kitchen | balcony_side_wall | 466 | 230 | −50.6 | fail (fragmented) |
| hall_kitchen | tv_side_wall | 469 | 207 | −56.0 | fail (fragmented) |
| hall_kitchen | bedroom_side_wall | 469 | 195 | −58.5 | fail (unswept wall) |

1/4 walls within ±3% (was 0/4 pre-fix; `kitchen_side_wall` moved +15.6% → +0.3%). The three long walls
are fragmented because the capture path did not sweep them face-on — a coverage limit, not a bug.

### Photo tier (gate: ±8%)

| Room | Wall | Tape (cm) | Measured (cm) | Error (%) | Pass? |
|---|---|---|---|---|---|
| hall_kitchen | kitchen_side_wall | 526 | PENDING | — | — |
| hall_kitchen | balcony_side_wall | 466 | PENDING | — | — |
| master_bedroom_b2 | width | 361 | PENDING | — | — |
| master_bedroom_b2 | length | 301.5 | PENDING | — | — |
| bedroom_b1 | length | 253.5 | PENDING | — | — |
| bedroom_b1 | width | 241.5 | PENDING | — | — |

---

## Repeatability — H1 vs H2 (≤1 cm or 0.5%/wall across two captures of same room)

Two captures of the same hall (`hall_kitchen`), video tier, post-fix. *"Two captures of the same room at
the same tier agree within 1 cm or 0.5% per wall. Same room in, same plan out is what 'spit out the same
results' means operationally."* Walls matched to the four tape walls independently in each run.

| Tier | Room | Wall | Tape | H1 (cm) | H2 (cm) | Diff (cm) | Diff (%) | Pass? |
|---|---|---|---|---|---|---|---|---|
| Video | hall_kitchen | kitchen_side_wall | 526 | 527 | 262 | 266 | 50.4% | fail |
| Video | hall_kitchen | tv_side_wall | 469 | 207 | 315 | 108 | 52.4% | fail |
| Video | hall_kitchen | bedroom_side_wall | 469 | 195 | 208 | 13 | 6.8% | fail |
| Video | hall_kitchen | balcony_side_wall | 466 | 230 | 327 | 97 | 42.1% | fail |

| Metric | H1 | H2 | Tape |
|---|---|---|---|
| Ceiling height | not observed | 288.7 cm | 279 |
| Floor area | 23.2 m² | 19.7 m² | ≈ 24 |
| Scale source | depth model (ruler rejected) | **reference ruler** | — |
| Runtime (cold) | 13.6 min | 9.0 min | — |

**Repeatability: 0/4 walls within gate — FAIL.** The two captures do not produce the same plan. Root
causes, in order of impact:
1. **Different scale source.** H1's ruler triangulation was implausible (see fix loop / limitations) so H1
   fell back to the depth-model scale; H2's ruler was accepted. Two different metric bases (H1 area 23.2 vs
   H2 19.7 m², ~15% apart) cannot repeat. Making ruler acceptance consistent is the top repeatability fix.
2. **Different coverage.** Each handheld pass sweeps different walls face-on, so the fragmented walls (all
   but `kitchen_side`) match different physical walls between runs — the 50%+ disagreements are the
   assignment pairing a full wall in one run with a fragment in the other.
3. H2 observed the ceiling (288.7 cm, +9.7 cm vs tape); H1 did not — the bowed-ceiling detection is itself
   run-dependent.

This is the honest operational answer: **same room in, same plan out does not hold yet on video.** The
single highest-value repeatability fix is consistent metric scale (ruler), tracked as an open item.

*Regenerate: `uv run floorfathom plan Data/H2.MOV --out out/h2 --reference-length-cm 31.6` (H1 in `out/h1`).*

Photo and LiDAR have no repeat capture. For LiDAR see "Not done: needs an iPhone Pro" below, which reports
the estimator-only tests we ran instead.

---

## Drift ablation (LiDAR only)

| Scan | Footprint ON (m²) | Footprint OFF (m²) | Max shift (m) | Max yaw (°) |
|---|---|---|---|---|
| single_room | 14.2 (2 rooms) | 20.6 (3 rooms) | 0.10 | 1.5 |
| single_scan_floor_only | 51.3 (5 rooms) | 52.3 (6 rooms) | 0.22 | 3.2 |
| single_scan_with_ceiling | 58.0 (6 rooms) | 58.4 (6 rooms) | 0.16 | 3.7 |

*Values are in `plan.json` → `stitching.drift` for each LiDAR run (current code, 2026-09-21).*

With the correction on, the footprint changes by −30.9%, −1.9% and −0.7% relative to the uncorrected one, and a room
disappears in two scans (3 → 2 and 6 → 5). There is no tape truth for these scans, so this does not show
that the correction is more accurate. Evidence that it does what it should: on synthetic walks it recovers
injected drift of 14 cm and 0.7° to under 2 cm (`tests/test_drift.py`), and the registration disagreement
of the sample scans falls by about 95%. Evidence against a clear gain: on two different recordings of the
same flat (`single_scan_with_ceiling` vs `single_scan_floor_only`, `scripts/cross_scan_repeatability.py
[--drift]`, prior 2 cm / 0.2°, measured at commit `19f8592`) the correction did not make the rooms agree
better: 6 vs 5 matched rooms, mean IoU 0.77 vs 0.78, summed area difference 7.2 vs 10.9 m², median wall
difference 44 vs 55 cm. That test is weak because the two walks did not see the same things. The
correction stays on by default because the recorded poses must not be trusted as they are on a long walk.

---

## Photo whole-property stitch

| Metric | Result | Gate | Pass? |
|---|---|---|---|
| All rooms placed | PENDING | All rooms in one plan | — |
| Correct adjacency | PENDING | No room overlaps | — |
| Footprint vs tape | PENDING | ±8% | — |

---

## Timing

| Tier | Capture | Cold (min) | Cached (min) |
|---|---|---|---|
| LiDAR | single_room | 2.0 | 2.0 |
| LiDAR | single_scan_floor_only | 4.2 | 4.2 |
| LiDAR | single_scan_with_ceiling | 7.2 | 7.2 |
| Video | H1.MOV (13 s clip) | 13.6 | 0.7 (44.6 s, post-fix, cached) |
| Video | H2.MOV | 9.0 | — |
| Video | B1.MOV | PENDING | — |
| Video | B2.MOV | PENDING | — |
| Photo | Data/ (all rooms) | 4.0 | 2.3 |

LiDAR uses no caches, so cold and cached are the same. Per-stage seconds are written to `plan.json` →
`diagnostics.notes`; for `single_scan_with_ceiling`: cloud 15, drift 5, estimate 4, bootstrap 74,
stitching 6, damage 319 s (431 s in all).

---

## Not done: needs an iPhone Pro (LiDAR)

No iPhone Pro (LiDAR scanner) was available to us. Everything we captured ourselves (`Data/`) is photo and
video from an iPhone 16. The LiDAR tier runs on the Cozmo-provided sample scans in `CozmoData/`, a flat we
cannot visit or measure, so it has no tape truth. Three required items depend on a LiDAR capture of our own
and were not done as specified.

### Head-to-head against a consumer scanning app (Part 3): NOT RUN

The brief asks for the LiDAR-tier output against one consumer scanning app on the same two rooms, error
against tape, dimension by dimension. We could not capture a LiDAR scan of any tape-measured room, and we
cannot run a consumer app inside the Cozmo sample flat. A comparison on the sample scans could show how an
app's export and our plan agree with each other, but not which one is closer to the truth, so it would not be
an error table and we do not present one as such. No app name or version is given because no app was run.

| Dimension | Ours (error vs tape) | App (error vs tape) | Tape |
|---|---|---|---|
| all | NOT RUN | NOT RUN | — |

### LiDAR repeatability (1 cm or 0.5% per wall; ceiling spread ≤1 cm): NOT RUN as specified

The gate compares two captures of the same room at the same tier. We have one LiDAR scan per flat, so there
is no repeat capture, and no ceiling-spread comparison either (`single_scan_floor_only` observes no
ceiling). What we measured instead tests the estimator, not the device, involves no tape truth and does not
replace the gate (measured at commit `19f8592`):

| Test | Result |
|---|---|
| Same walk, two disjoint frame subsets (`scripts/same_walk_repeatability.py`, `single_scan_with_ceiling`, six rooms match one to one) | Room areas within 1% for four rooms and 5% for the two smallest. Wall lines: median position difference 0.3 cm (76% within 1 cm). Corners with both neighbours supported: median 1.5 cm (46% within 1 cm). A wall end next to an unsupported edge (doorway or unseen stretch): median 31 cm. Only 26% of the walls lying on the same line in both runs meet 1 cm / 0.5% (median length difference 9 cm, n = 19). With drift correction on, lines move 1.8 cm and corners 3.2 cm, because each run estimates the drift separately (the two estimates differ by a median 1.5 cm, up to 3.5 cm and 0.9°). |
| Two different recordings of the same flat (`scripts/cross_scan_repeatability.py`, `single_scan_with_ceiling` vs `single_scan_floor_only`) | All six rooms match one to one (overlap 0.71–0.90). The largest room agrees to 0.3% in area, the others differ by 4–27%; median wall length difference 44 cm. The two walks did not see the same things, so this does not isolate the estimator. |

Verdict: the gate is not met. Wall lines repeat well; wall lengths fail because of where a wall ends
(segmentation next to doorways and unseen stretches), not because the wall moves.

### LiDAR on tape-measured rooms; the same rooms at all three tiers

For the same reason the LiDAR tier was never run on the rooms in `ground_truth.json`. "The same rooms at all
three input tiers" therefore holds for photo and video only, LiDAR accuracy against tape is unmeasured, and
the LiDAR capture protocol was never tested on a live capture (`capture_protocol.md`).
