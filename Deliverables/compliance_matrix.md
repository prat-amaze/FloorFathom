# Compliance Matrix

**Format:** Requirement → File path → Artifact → Status
**Status key:** PASS | PARTIAL | FAIL | NOT RUN | PENDING

---

## Part 1 — Capture route and tiers

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Route 2 stock capture protocol (one page, non-engineer follows literally) | `capture_protocol.md` | Full protocol with install, walk, handoff sections | PASS |
| Device matrix (which tier on which hardware, honest accuracy per tier) | `capture_protocol.md` | Device matrix table | PASS |
| Photo tier — 2-8 stills per room, any iPhone 15+, no depth, no poses | `src/floorfathom/photo_pipeline.py` | `run_photo()` entry point | PASS |
| Video tier — handheld walkthrough clip, any iPhone 15+ | `src/floorfathom/video_pipeline.py` | `run_clip()` / `run_video()` entry points | PASS |
| LiDAR tier — depth, poses, intrinsics, Pro-class device | `src/floorfathom/pipeline.py` | `run_lidar()` entry point | PARTIAL — tier runs on Cozmo sample data; no Pro device available for live capture |
| All three tiers mandatory (same output contract from each) | `src/floorfathom/cli.py` | Single `floorfathom plan` command dispatches by tier | PASS |

---

## Part 2 — Output contract (per capture, all tiers)

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Dimensioned per-room walls | `schema/capture_plan.schema.json` | `rooms[].walls[].length` (value, lo, hi, unit) | PASS |
| Ceiling height per room | `schema/capture_plan.schema.json` | `rooms[].ceiling_height` (Measurement) | PASS |
| Floor area per room | `schema/capture_plan.schema.json` | `rooms[].floor_area` (Measurement) | PARTIAL — null in photo tier (incomplete outline pre-SfM rewrite) |
| Openings with detection and width | `schema/capture_plan.schema.json` | `rooms[].openings[].width` (Measurement) | PARTIAL — video: widths found but outside ±2 cm gate; photo: not scored on real run |
| Stitched multi-room plan with correct adjacency | `schema/capture_plan.schema.json` | `stitching.links`, `stitching.placements`, `stitching.footprint` | PARTIAL — video B1 stays in own frame; photo adjacency incomplete |
| Per-surface damage regions with class and metric extent | `schema/capture_plan.schema.json` | `rooms[].damage[].damage_class`, `.width`, `.height` (each a Measurement with lo/hi) | PASS |
| Concealed-damage flags with the rule that fired | `schema/capture_plan.schema.json`, `src/floorfathom/concealed.py` | `rooms[].concealed_flags[].rule` (named rule per flag) | PASS |
| Scope line items keyed to surfaces | `schema/capture_plan.schema.json`, `src/floorfathom/scope.py` | `rooms[].scope[].surface`, `.action`, `.quantity` | PASS |
| Confidence interval on every measurement | `src/floorfathom/uncertainty.py`, `schema/capture_plan.schema.json` | Every `Measurement` has `.lo` and `.hi` (95% CI) | PASS |
| One command per capture | `src/floorfathom/cli.py` | `uv run floorfathom plan <capture> --out <dir>` | PASS |
| JSON to published schema | `schema/capture_plan.schema.json` | Schema version 0.4.0; test checks code matches file | PASS |
| Rendered plan | `src/floorfathom/render.py` | `out/<name>/plan.png` — top-down with dimensions, doorways, damage | PASS |

---

## Part 2 — Accuracy gates

| Gate | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| Opening widths | ≤2 cm error on ≥85% of openings; missed/phantom = miss | `Deliverables/benchmark_report.md` | Gate table with per-opening errors | FAIL (video: all 6 found, 1/6 within 2 cm; photo not scored; LiDAR NOT RUN, no tape truth — see benchmark report) |
| Ceiling height | ≤1.5 cm per room; spread ≤1 cm across captures | `Deliverables/benchmark_report.md` | Gate table with per-room errors | FAIL (video H1 +6.5 cm; photo NOT RUN post-rewrite — see benchmark report) |
| Repeatability | Two captures agree ≤1 cm or 0.5%/wall | `Deliverables/benchmark_report.md` | Repeatability table (H1 vs H2 video) | FAIL (video H1 vs H2: 0/4 walls within the gate). LiDAR NOT RUN as specified: one scan per flat and no iPhone Pro to make a repeat capture; the estimator-only tests in the report do not meet the gate either. Photo: no repeat capture |
| Drift accountability | Report states drift handling; ablation shows footprint with/without | `out/<name>/plan.json` → `stitching.drift`, `Deliverables/technical_report.md` §3 | On/off footprint in every LiDAR plan.json; ablation numbers in benchmark report | PASS (LiDAR); N/A (video, photo) |
| Photo whole-property stitch | Per-room photo folders → one plan, correct adjacency, footprint ±8% | `Deliverables/benchmark_report.md` | Stitch table | FAIL (floor area null pre-SfM rewrite; re-evaluation pending) |

---

## Part 2 — Benchmark composition

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| One multi-room capture (≥3 rooms + connector) | `Data/H1.MOV`, `Data/B1.MOV`, `Data/B2.MOV`, photo folders `Data/Hall`, `Data/B1`, `Data/B2`, `Data/Full.MOV` | hall_kitchen (connector) + bedroom_b1 + master_bedroom_b2 captured; the bathroom has tape truth in `ground_truth.json` but no capture is present in `Data/` | PARTIAL |
| One furnished room with staged damage spanning 2 damage classes | `Data/H1.MOV`, `ground_truth.json` | hall_kitchen: water_damage (leakage mark) + structural_crack (kitchen crack) | PASS |
| Same rooms captured at all 3 tiers (multi-room set included) | `Data/` (photo + video), `CozmoData/` (LiDAR) | Photo + video share the same flat; LiDAR uses Cozmo sample data (different property) | PARTIAL — photo and video only; LiDAR was not captured on these rooms (no iPhone Pro), see benchmark report "Not done" |
| At least one room captured twice at same tier (repeatability) | `Data/H1.MOV`, `Data/H2.MOV` | Two Hall video takes, compared in the benchmark report (0/4 walls within the gate) | PASS for video (captured and compared); no repeat capture at photo or LiDAR |
| Tape ground truth on everything; raw sensor data submitted | `ground_truth.json`, `Data/`, `CozmoData/` | ground_truth.json covers all 4 Data/ rooms (walls, ceiling, openings, damage); CozmoData/ has no tape truth | PARTIAL — CozmoData/ unmeasured (no Pro device, sample data only) |

---

## Part 3 — Head-to-head vs consumer app

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| 2 benchmark rooms, LiDAR tier output vs named consumer app on same rooms | `Deliverables/benchmark_report.md` | "Not done: needs an iPhone Pro" section | FAIL — NOT RUN: no iPhone Pro, so no LiDAR capture of a tape-measured room and no consumer-app comparison |
| One table: your error vs theirs, dimension by dimension | `Deliverables/benchmark_report.md` | Table stub marked NOT RUN | FAIL — NOT RUN |
| Beat or tie on ≥70% of shared dimensions | `Deliverables/benchmark_report.md` | — | FAIL — NOT RUN |

---

## Part 4 — Fix loop

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Fix declaration: worst gate, failing number, root cause, predicted after | `Deliverables/fix_loop/declaration.md` | Video opening detection on H1 (1/6 found), root cause with evidence, fix, predicted and actual numbers | PARTIAL — written, but first committed in `7055a33`, the same commit as the fix, so the git history does not show the prediction preceding the fix |
| Shipped fix | `src/floorfathom/video_heights.py` (`floor_from_strongest_spike`), `src/floorfathom/video_pipeline.py`, `tests/test_video_heights.py` | Landed in `7055a33`, a 34-file commit that also holds photo-tier work; the fix itself is 3 files, +106/-2 lines | PASS |
| Before run (regenerable) | `Deliverables/fix_loop/declaration.md` (command) | `out/fix_loop_before/` is not in the repository (`out/` is git-ignored); regenerate with the declaration's command, about 13 min | PENDING — run to be captured |
| After run (regenerable) | `Deliverables/fix_loop/declaration.md` (command) | `out/fix_loop_after/`, same command | PENDING — run to be captured |
| Readable diff | git | `git diff 7055a33^ 7055a33 -- src/floorfathom/video_heights.py src/floorfathom/video_pipeline.py tests/test_video_heights.py` | PASS |

---

## Part 5 — Process evidence

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Commit history grows with work (not one- or two-commit materialisation) | git log | Ongoing commit history across development | PASS |

---

## Deliverables checklist

| # | Deliverable | File path | Status |
|---|---|---|---|
| 1 | Compliance matrix | `Deliverables/compliance_matrix.md` | PASS |
| 2 | Capture route + device matrix | `capture_protocol.md` | PASS |
| 3 | Repo + README (≤15 min setup, 1 command per capture) | `README.md` | PASS |
| 4 | Reproduction bundle (raw inputs regenerate every number; live path runs) | `Data/`, `CozmoData/`, `ground_truth.json`; weights via `uv run floorfathom fetch-models` | PARTIAL — deterministic for LiDAR and photo; video has run-to-run variation |
| 5 | Benchmark report (gates at all 3 tiers, repeatability table, timing) | `Deliverables/benchmark_report.md` | PARTIAL — video H1/H2 tables, LiDAR timing and drift ablation, and the "Not done: needs an iPhone Pro" section are filled; photo wall lengths and stitch and video B1/B2 rows are pending |
| 6 | Fix loop bundle | `Deliverables/fix_loop/` | PARTIAL — declaration written, fix shipped, diff command given; before/after runs still to be regenerated |
| 7 | Technical report (max 6 pages) | `Deliverables/technical_report.md` | DRAFT — all sections filled; page count not checked against the 6-page cap |
| 8 | Raw benchmark data (sensor logs, ground truth) | `Data/`, `CozmoData/`, `ground_truth.json` | PARTIAL — ground_truth.json complete for Data/; CozmoData/ has no tape truth |

---

## Known gaps that need an iPhone Pro

No iPhone Pro (LiDAR scanner) was available to us. Consequences, each detailed in
`Deliverables/benchmark_report.md` ("Not done: needs an iPhone Pro"): no head-to-head against a consumer
app; no LiDAR repeatability gate (no repeat LiDAR capture); the LiDAR tier was never run on the tape-measured
rooms, so "the same rooms at all three tiers" holds for photo and video only; the LiDAR capture protocol was
never tested on a live capture.

---

## Constraints

| Constraint | File path | Artifact | Status |
|---|---|---|---|
| Handheld consumer capture only | `capture_protocol.md` | iPhone native Camera app; no custom iOS app built | PASS |
| Any pretrained model with disclosure | `README.md`, `src/floorfathom/models.py` | Depth Anything V2 Metric-Indoor (HuggingFace); pycolmap (open source) | PASS |
| Everything runs without calling your infrastructure | `src/floorfathom/` | All computation local; no external API calls at runtime | PASS |
| Weights and large binaries fetched by script | `src/floorfathom/models.py` | `uv run floorfathom fetch-models` downloads from HuggingFace | PASS |
| Mirrors, glass, wet-look surfaces, low light covered in submission | `edge_cases.md`, `capture_protocol.md`, `Deliverables/technical_report.md` §6 | edge_cases.md lists mirrors/glossy floor; protocol advises avoiding glass/mirrors; technical report §6 documents failure modes | PARTIAL — covered in docs; no dedicated test on low-light captures |

---

## Walk-in test readiness

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| All three tiers ready to run on a fresh capture | `src/floorfathom/cli.py` | Single command per tier; no manual steps | PARTIAL — photo and video ready; LiDAR ready against sample data format but untested on live Pro-device capture |
| Pipeline runs cold (no pre-cached state needed) | `src/floorfathom/` | Caches are optional; cold path runs end to end | PASS |
| Capture route followed literally produces a runnable capture | `capture_protocol.md` | Protocol tested on iPhone 16 non-Pro for photo + video tiers | PARTIAL — LiDAR protocol not live-tested (no Pro device) |
