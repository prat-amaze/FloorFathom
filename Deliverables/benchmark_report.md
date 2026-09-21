# Benchmark Report

Gate numbers, repeatability table, timing. Video tier numbers are the **post-fix** runs
(commit `a0056a2`, glossy-floor reflection fix — see `fix_loop/declaration.md`). All video
runs used `--reference-length-cm 31.6`.

---

## Photo tier — gate summary and root causes (read this first)

The photo tier **fails all five accuracy gates** on the real four-room benchmark. This is expected for
the PDF's designated "floor" tier (2–8 handheld stills, no depth, no poses) and the failures are honest,
not confident garbage — nulls, wide-but-tape-covering intervals and flags, never a wrong confident number.
Each failure traces to one of three causes, only one of which is a code lever:

| Gate | Result | Root cause | Category | Would pass if… |
|---|---|---|---|---|
| Wall ±8% | 0/4 rooms | outlines never close (walls are fragments) | **capture** | dense enough overlap that SfM registers most frames and every wall is swept |
| Floor area | null everywhere | open outline → no polygon | **capture** | same as walls |
| Openings ≤2 cm | 0/9 detected | no doorway-width through-wall gaps in an open outline | **capture** | same as walls |
| Whole-property stitch ±8% | 1/4 placed, 0 links | 0 doorways → nothing to glue | **capture** (stitch code is sound, tests pass) | rooms close outlines and expose doorways |
| Repeatability (Hall/Hall2) | fail (ceiling 30% apart) | inconsistent scale source (ruler found on Hall, not Hall2) | **capture + ruler discipline** | the ruler is captured in ≥3 photos in *both* takes |
| Ceiling ±1.5 cm | fail (+25 cm Hall) | (a) monocular ceiling bowed/smeared over ~60 cm; (b) estimator picks the highest plane, ~+25 cm high | **depth-limited + method** | not achievable at ±1.5 cm with monocular depth at *any* handheld tier (LiDAR only) |

**Why SfM under-registers (the master cause):** covering a room with only 6–9 photos means each photo
faces a different wall, so consecutive photos share little 3-D structure even when they share features.
SfM links only the densely-overlapping subset (measured: Hall 4/8, Hall2 4/9, B1 0/6, B2 7/10 but
mirror-doubled). Camera mode (SINGLE vs PER_IMAGE) makes no difference — confirmed, so it is coverage,
not a solver setting. Below 60% registration the tier falls back to a single-station cloud that cannot
see every wall → open outline → the whole cascade above. The original rotation-only photo tier failed
these same gates for the same reason; this is a property of monocular photo reconstruction of whole
rooms, not a regression.

**The one code lever identified (not shipped):** on the ruler-scaled Hall the ceiling band is bimodally
bowed — a near-ceiling plateau at 2.4 m and a far spike at 3.0 m, with the true 2.79 m in the trough
between. `photo_layout.floor_and_ceiling` picks the highest area-like plane (3.0 m, +25 cm); the *mean*
of the band is 2.74 m (−5 cm). A band-centre ceiling would cut the bias from +25 cm to ~−5 cm — a real
improvement, **but it still fails the ±1.5 cm gate** and is validated on only one room (Hall2/B1 ceilings
are scale-dominated, not detection-dominated), so shipping it would be over-fitting to a single sample
against the PDF's own four-room-can't-calibrate warning. Recorded as a candidate, deliberately not shipped.

**Bottom line:** the photo gates are not passable on 6–9 handheld stills of whole rooms with monocular
reconstruction; the honest path to passing them is capture (denser overlap + ruler discipline), which is
in `capture_protocol.md`. The tier's job at this input thinness is to degrade honestly (nulls, wide
intervals, flags), which it does — the behaviour the PDF's calibration scoring rewards over confident garbage.

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
| Photo | all (Hall/Hall2/B1/B2) | 9 unique | **0 detected** | — | **fail — 0/9 (every tape opening missed)** |

Detection (video): **6/6 openings found** post-fix vs **1/6 pre-fix**. Positions are biased wide because the
three interior doors sit on a sheared room outline (unswept left wall); see the fix loop post-mortem.

Detection (photo): **0 of 9 openings detected across all four rooms** → fail (a missed opening counts as a
miss). No phantom openings either. Cause is the same as the stitch failure: the rooms reconstruct as open
outlines with no doorway-width through-wall gaps, so `find_openings` has nothing to report (diagnosed in
the whole-property stitch section — the Hall's only wall gap is 2.0 m, above the 1.8 m doorway max). Fixable
only by capture coverage, not detection tuning.

---

## Ceiling height (≤1.5 cm per room; spread ≤1 cm across captures)

| Tier | Room | Tape (cm) | Measured (cm) | Error (cm) | Pass? |
|---|---|---|---|---|---|
| LiDAR | single_scan_with_ceiling (6 rooms) | no tape truth | 307.6, 307.5, 241.8, 306.4, 234.6, 227.9 (each ±1.7–1.8 cm) | — | not scorable: no tape truth on CozmoData; the other two scans observe no ceiling |
| Video | hall_kitchen (H1) | 279 | not observed | — | fail (not reported rather than guessed) |
| Video | hall_kitchen (H2) | 279 | 288.7 | +9.7 | fail |
| Video | bedroom_b1 | 279 | not observed | — | fail |
| Video | master_bedroom_b2 | 279 | 263.8 | −15.2 | fail |
| Photo | Hall (hall_kitchen) | 279 | 303.7 | +24.7 | fail (tape inside interval [259, 348]; ruler scale) |
| Photo | Hall2 (hall_kitchen) | 279 | 396.9 | +117.9 | fail (tape inside wide interval [160, 634]; no ruler) |
| Photo | bedroom_b1 | 279 | 380.9 | +101.9 | fail (tape inside wide interval [153, 608]; no ruler) |
| Photo | master_bedroom_b2 | 279 | not observed | — | fail (ceiling implausible; mirror room) |

Ceiling is observed on 2 of 4 video captures and 3 of 4 photo captures, never within the 1.5 cm gate.
On the video H1/B1 and the photo B2 it is reported as not-observed rather than a wrong number: the
depth-model ceiling is bowed and smeared 1.9–3.4 m above the floor with competing spikes, and the room
mask it would be validated against is mis-shaped. The photo ceilings are biased high (+25 to +118 cm)
and only the ruler-scaled Hall is close; the three no-ruler rooms carry very wide intervals that still
contain the tape. Distinct from the shipped reflection fix — a known open item at both tiers.

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

### Video tier (gate: ±3%), bedroom_b1 (B1) and master_bedroom_b2 (B2), post-fix

| Room | Wall (tape cm) | Measured (cm) | Error (%) | Pass? |
|---|---|---|---|---|
| bedroom_b1 | 253.5 | 286 | +12.9 | fail |
| bedroom_b1 | 241.5 | 166 | −31.4 | fail |
| bedroom_b1 | 253.5 | 243 | −4.1 | fail |
| bedroom_b1 | 241.5 | 179 | −25.9 | fail |
| master_bedroom_b2 | 361 | 374 | +3.7 | fail |
| master_bedroom_b2 | 301.5 | 317 | +5.3 | fail |
| master_bedroom_b2 | 361 | 352 | −2.4 | **PASS** |
| master_bedroom_b2 | 301.5 | 293 | −3.0 | **PASS** |

B2 (a small, well-covered rectangular room) is the best video result: **2/4 walls within ±3%** and floor
area **11.04 m² vs tape 10.88 (+1.5%)**. B1 is the worst (0/4, area 5.30 vs 6.12 m², −13%) — a small room
where the glossy floor and few features gave a sparse cloud and no ceiling.

**Video wall summary: 3 of 16 walls within ±3% across all four captures (H1 1/4, H2 0/4, B1 0/4, B2 2/4).**
Accuracy tracks capture coverage: B2 (best-covered) 2/4, the hall (large, diagonally-walked) 0–1/4.

### Photo tier (gate: ±8%)

Real run, four photo rooms, `--reference-length-cm 31.6` (Hall in `out/hall_fixed`, the rest in
`out/<room>_bench`). Only the Hall found a usable ruler; the other three fell back to the depth-model
scale. **No room produced a closed outline**, so every wall is a fragment of a longer wall and floor
area is null everywhere (see whole-property stitch and limits). Measured lengths are listed as reported;
a fragment is not matched 1:1 to a tape wall because it is not the wall.

| Room | Tape walls (cm) | Photo walls measured (cm) | Within ±8%? | Notes |
|---|---|---|---|---|
| Hall (hall_kitchen) | 526, 466, 469, 469 | 237, 285, 268 (+1 closure) | 0/4 | fragments (~half length); ruler scale; SfM 4/8 → rotation-only |
| Hall2 (hall_kitchen) | 526, 466, 469, 469 | 183, 71, 433, 258 (+2 closure) | 0/4 | fragments; no ruler → depth-model scale; SfM 4/9 → rotation-only |
| B1 (bedroom_b1) | 253.5, 241.5 | 265, 54, 222, 258 (+1 closure) | 2 fragments land in ±8% (265→+4.4%, 258→+1.8%) but the outline is open, so not a valid gated wall | no ruler; SfM failed → rotation-only |
| B2 (master_bedroom_b2) | 361, 301.5 | 613, 934, 143, 418, 489, 483, 906 | 0/7 | full-length-mirror edge case: SfM registered but doubled/mis-scaled the room 2–3×; no ruler |

**Photo walls: 0 of 4 rooms pass ±8% with a valid (closed-outline) measurement.** The cause is
coverage/registration, not the estimator: the real captures are 6–9 wide-baseline stills that
under-register in SfM and fall back to a single-station cloud that cannot see every wall, so the
outline never closes. B2 additionally hits the mirror edge case, which lets SfM register but doubles
the apparent room. The fix is capture (denser, higher-overlap photos, ruler in ≥3 frames), per
`capture_protocol.md`.

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

### Photo tier — Hall vs Hall2 (two photo captures of the same hall)

Two separate photo folders of the same hall (`hall_kitchen`), photo tier. Both under-registered in
SfM and fell back to the single-station path, so neither closed the outline — the walls are fragments
that cannot be matched 1:1 between runs. The comparable repeatable quantity is the ceiling.

| Metric | Hall | Hall2 | Tape | Agree? |
|---|---|---|---|---|
| Ceiling height | 303.7 cm | 396.9 cm | 279 | **no** — 93.2 cm / 30.7% apart |
| Floor area | null (open outline) | null (open outline) | ≈ hall | both null |
| Scale source | **reference ruler** | depth model (no ruler found) | — | different |
| Walls | 3 fragments | 4 fragments | 526/466/469/469 | not matchable (both open outlines) |
| SfM registration | 4/8 → rotation-only | 4/9 → rotation-only | — | both fell back |

**Photo repeatability: FAIL — same room in, different plan out.** Same root cause as the video H1/H2
pair: the two captures took a **different scale source** (Hall's ruler was accepted, Hall2's was never
found), and each single-station cloud sees different walls, so the ceiling disagrees by 30% and the
walls are unmatchable fragments. Consistent ruler capture (ruler in ≥3 well-separated photos, per
`capture_protocol.md`) is the top repeatability fix for the photo tier, as for video.

*Regenerate: `uv run floorfathom plan Data/Hall --tier photo --out out/hall_fixed --reference-length-cm 31.6`
and `Data/Hall2 --out out/Hall2_bench`.*

---

## Video whole-property stitch

Folder of one-room clips → one stitched plan: `uv run floorfathom plan out/clips --out out/video
--reference-length-cm 31.6` (`out/clips` = H1, B1, B2; cache-reused, 213 s).

| Metric | Result | Gate | Pass? |
|---|---|---|---|
| Rooms placed | 2 of 3 (hall H1 + master_bedroom B2) | all rooms in one plan | PARTIAL |
| Adjacency | H1 ↔ B2 glued via mutual doorway, gap 0.15 m | correct adjacency, no overlaps | PASS (matches tape: hall ↔ b2_door ↔ master_bedroom) |
| Unplaced | B1 (bedroom) — no doorway observed linking it | — | honest (kept in own frame, not forced) |
| Footprint | 34.28 m² [9.16, 59.41] | — | wide interval (per-room outlines are rough) |

The video tier **does** stitch a multi-room plan with correct adjacency: master_bedroom_b2 connects to
the hall through the b2 doorway (mutual link, gap 0.15 m — the assumed wall thickness), which is the true
adjacency in `ground_truth.json`. B1 is honestly left unplaced because no doorway crossing to it was
observed in its clip. The footprint interval is wide because the underlying per-room outlines are rough
(see wall section).

**Whole-flat single clip (`Data/Full.MOV`).** As a stress test, the entire flat walked as one 107 s clip
was run as a single capture: `uv run floorfathom plan Data/Full.MOV --out out/full --reference-length-cm 31.6`
(cold, 21.8 min, ~530 keyframes). A single clip is one room by design, but SfM split the long walk into
several models (`sfm_split_into_several_models`), so the pipeline recovered **2 partial rooms** (24.9 m²,
ceiling 291.5 cm; 6.5 m², ceiling 305.6 cm — both ceilings observed) and stitched them (footprint 31.4 m²,
2 mutual doorway links, none unplaced). It does **not** recover all four rooms — the supported multi-room
path is the folder-of-one-clip-per-room command above; `Full.MOV` degrades gracefully (2 stitched rooms)
rather than crashing.

**Video drift accountability.** Each clip is reconstructed in its own SfM frame; there is no pose chain
between rooms, so inter-room drift cannot accumulate — placement error enters only through the doorway
position and the assumed wall thickness. This is stated in `out/video/plan.json` → `stitching.drift`.
Ablation (footprint with/without correction) applies to the LiDAR tier's pose-graph drift, not to the
video tier, which has no cross-room pose chain to correct. Drift *within* one clip's walk is handled by
SfM bundle adjustment, not a separate pass.

---

## Test suite

`uv run pytest tests/` — **229 passed, 15 skipped, 1 failed** (2026-09-21; confirmed again this session,
571 s under concurrent load). The single failure is
`test_lidar_tier.py::test_pipeline_output_is_valid_deterministic_and_covers_truth`.

- **Root cause of the failure (LiDAR):** the test runs the LiDAR pipeline twice with the same seed and
  asserts byte-identical JSON, but the JSON embeds per-stage wall-clock **timing** in `diagnostics.notes`
  (e.g. `cloud 4, …, stitching 2` vs `cloud 3, …, stitching 3`). Those seconds vary with machine load, so
  the test is flaky under load (it failed here while the photo/video SfM runs shared the CPU). The geometry
  is deterministic; only the timing string differs. It is a test-design issue (timing should be excluded
  from the determinism comparison), not a pipeline regression, and is unrelated to the photo tier.
- **Photo tier tests: all pass** — `test_photo_pose` (SfM registration + EXIF-focal fallback),
  `test_photo_scene`, `test_photo_layout_manhattan`, `test_photo_reference`, `test_photo_pipeline_integration`
  (end-to-end synthetic walk), `test_stitch_refine`, `test_eval_photo_geometry`, `test_synth_photo`,
  `test_photo_damage`. The two superseded rotation-only files (`test_photo_pipeline`, `test_photo_frames`)
  are skipped (replaced by the SfM-interface tests) — part of the 15 skips.
- **Video tier tests: all pass** (`test_video_heights`, `test_video_rays`, `test_video_openings`,
  `test_video_pipeline`, `test_points_video`, `test_damage_video`, `test_eval_video`).

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

Run: `uv run floorfathom plan Data --tier photo --out out/photo --reference-length-cm 31.6` (4 rooms:
B1, B2, Hall, Hall2; 194 s cold). Result in `out/photo/plan.json`.

| Metric | Result | Gate | Pass? |
|---|---|---|---|
| All rooms placed | 1 of 4 (B1 = root; B2, Hall, Hall2 unplaced) | All rooms in one plan | fail |
| Correct adjacency | 0 links — no doorway detected on any room, so nothing to glue | No room overlaps | fail |
| Footprint vs tape | 8.59 m² but it is only the root room's open outline (not a property footprint) | ±8% | fail |

**Photo stitch: FAIL — 1/4 rooms placed, 0 adjacency links.** Stitching needs a detected doorway on two
rooms to glue them; on these captures no room closed its outline or detected an opening (0 openings across
all four rooms), so three rooms are reported `unplaced` and stay in their own frames.

**Why (diagnosed, `scripts/diagnose_photo_openings.py` on the Hall — which physically has 3 doorways to
B1, B2 and the bathroom):** the Hall reconstructs as **4 wall fragments, outline open (not closed)**; the
only gap in any wall is 2.0 m wide, above the 0.55–1.8 m doorway window, so it reads as an unobserved wall
stretch, not a door. The three real 71–81 cm doorways are **not present in the reconstructed geometry** —
the walls that contain them are unreconstructed or came out as single runs without the door gap. So there
is nothing to glue: this is a capture/reconstruction limit, not a stitching bug.

`stitch.py` itself is verified sound for proper input: `tests/test_stitch.py` places a hub + three rooms
by their doorways with correct adjacency and **zero overlap**, reports unplaceable rooms rather than
guessing, and counts the footprint with overlaps once; `tests/test_stitch_refine.py` covers the joint
`refine_global` pass. The tier is deliberately **not** made to force placements on undetected doorways —
that would be the "confident garbage on thin input" the case study caps the score for. The fix is capture:
denser, higher-overlap photos that close each room's outline and expose the door gaps (`capture_protocol.md`).

---

## Damage (verified benchmark accuracy)

Photo tier, `out/photo/plan.json`, scored by `scripts/eval_damage_photo.py` against the two staged items
in `ground_truth.json` (Hall). The damage stage runs automatically as part of the photo command.

| Item | Tape (cm) | Photo result | Class | Pass? |
|---|---|---|---|---|
| leakage_mark (water_damage) | 21 × 25 | **found** on Hall wall, 17 × 26 cm [14.5–20.6 × 22.0–30.1] | water_stain ✓ | height inside interval; width 17 vs 21 short (outside) |
| kitchen_crack (structural_crack) | 12 × 22 | **not found** (hairline crack, not resolvable in stills) | — | miss |

**Photo damage: 1 of 2 staged items found (class correct), 10 false-positive regions** across the four
rooms (B1: 3 cracks on wood grain + 2 soot on wardrobe/wall; Hall: 2 extra water stains; Hall2: 3 cracks).
The rules-based detector recovers the large water stain with the correct class and one dimension inside
interval, but the hairline crack is below the resolvable limit in stills, and plain wood grain / wardrobe
edges drive false positives (a known limit shared with the video tier; a CLIP classifier was tried and
dropped — worse than the rules on real patches).

---

## Timing

| Tier | Capture | Cold (min) | Cached (min) |
|---|---|---|---|
| LiDAR | single_room | 2.0 | 2.0 |
| LiDAR | single_scan_floor_only | 4.2 | 4.2 |
| LiDAR | single_scan_with_ceiling | 7.2 | 7.2 |
| Video | H1.MOV (13 s clip) | 13.6 | 0.7 (44.6 s, post-fix, cached) |
| Video | H2.MOV | 9.0 | — |
| Video | B1.MOV | 8.7 | — |
| Video | B2.MOV | 12.3 | — |
| Video | Full.MOV (107 s whole-flat clip) | 21.8 | — |

*Video cold times measured this session on a loaded machine (three SfM jobs back-to-back): H2 540 s, B1 521 s,
B2 738 s, Full 1308 s. Areas: H1 23.2, H2 19.7, B1 5.30 (tape 6.12), B2 11.04 (tape 10.88) m². Full.MOV (~530
keyframes) SfM-split into 2 partial rooms — see whole-property stitch section.*
| Photo | Hall (8 stills) | 1.2 (73 s) | — |
| Photo | Hall2 (9 stills) | 1.1 (65 s) | — |
| Photo | B1 (6 stills) | 1.4 (83 s) | — |
| Photo | B2 (9 stills) | 2.4 (142 s) | — |

*Photo cold times measured this session (SfM + depth model, per room). Each room: SfM registration
(sequential + exhaustive matching), Depth Anything V2 densification, estimator and damage. B2 is slowest
(9 stills + the mirror room's larger, mis-scaled cloud).*

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
