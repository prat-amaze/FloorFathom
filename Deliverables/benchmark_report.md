# Benchmark Report

Gate numbers, repeatability table, timing. Video tier numbers are the **post-fix** runs
(commit `a0056a2`, glossy-floor reflection fix — see `fix_loop/declaration.md`). All video
runs used `--reference-length-cm 31.6`.

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
| LiDAR | — | — | — | — | NOT RUN (no tape truth on CozmoData) |
| Photo | — | — | — | — | see photo tier report |

Detection: **6/6 openings found** post-fix vs **1/6 pre-fix**. Positions are biased wide because the
three interior doors sit on a sheared room outline (unswept left wall); see the fix loop post-mortem.

---

## Ceiling height (≤1.5 cm per room; spread ≤1 cm across captures)

| Tier | Room | Tape (cm) | Measured (cm) | Error (cm) | Pass? |
|---|---|---|---|---|---|
| LiDAR | — | — | — | — | NOT RUN |
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

---

## Drift ablation (LiDAR only)

| Scan | Footprint ON (m²) | Footprint OFF (m²) | Max shift (m) | Max yaw (°) |
|---|---|---|---|---|
| single_room | PENDING | PENDING | PENDING | PENDING |
| single_scan_floor_only | PENDING | PENDING | PENDING | PENDING |
| single_scan_with_ceiling | PENDING | PENDING | PENDING | PENDING |

*Values are in `plan.json` → `stitching.drift` for each LiDAR run.*

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
