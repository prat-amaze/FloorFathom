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
| Per-surface damage regions with class and metric extent | `schema/capture_plan.schema.json` | `rooms[].damage[].damage_class`, `.width_m`, `.height_m` | PASS |
| Concealed-damage flags with the rule that fired | `schema/capture_plan.schema.json`, `src/floorfathom/concealed.py` | `rooms[].concealed_flags[].rule` (named rule per flag) | PASS |
| Scope line items keyed to surfaces | `schema/capture_plan.schema.json`, `src/floorfathom/scope.py` | `rooms[].scope[].surface_id`, `.action`, `.quantity` | PASS |
| Confidence interval on every measurement | `src/floorfathom/uncertainty.py`, `schema/capture_plan.schema.json` | Every `Measurement` has `.lo` and `.hi` (95% CI) | PASS |
| One command per capture | `src/floorfathom/cli.py` | `uv run floorfathom plan <capture> --out <dir>` | PASS |
| JSON to published schema | `schema/capture_plan.schema.json` | Schema version 0.4.0; test checks code matches file | PASS |
| Rendered plan | `src/floorfathom/render.py` | `out/<name>/plan.png` — top-down with dimensions, doorways, damage | PASS |

---

## Part 2 — Accuracy gates

| Gate | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| Opening widths | ≤2 cm error on ≥85% of openings; missed/phantom = miss | `Deliverables/benchmark_report.md` | Gate table with per-opening errors | FAIL (video 0/4; others NOT RUN — see benchmark report) |
| Ceiling height | ≤1.5 cm per room; spread ≤1 cm across captures | `Deliverables/benchmark_report.md` | Gate table with per-room errors | FAIL (video H1 +6.5 cm; photo NOT RUN post-rewrite — see benchmark report) |
| Repeatability | Two captures agree ≤1 cm or 0.5%/wall | `Deliverables/benchmark_report.md` | Repeatability table (H1 vs H2 video) | NOT RUN — H2.MOV exists; run pending |
| Drift accountability | Report states drift handling; ablation shows footprint with/without | `out/<name>/plan.json` → `stitching.drift`, `Deliverables/technical_report.md` §3 | On/off footprint in every LiDAR plan.json; ablation numbers in benchmark report | PASS (LiDAR); N/A (video, photo) |
| Photo whole-property stitch | Per-room photo folders → one plan, correct adjacency, footprint ±8% | `Deliverables/benchmark_report.md` | Stitch table | FAIL (floor area null pre-SfM rewrite; re-evaluation pending) |

---

## Part 2 — Benchmark composition

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| One multi-room capture (≥3 rooms + connector) | `Data/H1.MOV`, `Data/B1.MOV`, `Data/B2.MOV` | hall_kitchen (connector) + bedroom_b1 + master_bedroom_b2 + bathroom — 4 rooms | PASS |
| One furnished room with staged damage spanning 2 damage classes | `Data/H1.MOV`, `ground_truth.json` | hall_kitchen: water_damage (leakage mark) + structural_crack (kitchen crack) | PASS |
| Same rooms captured at all 3 tiers (multi-room set included) | `Data/` (photo + video), `CozmoData/` (LiDAR) | Photo + video share the same flat; LiDAR uses Cozmo sample data (different property) | PARTIAL — LiDAR not on the same rooms; no Pro device |
| At least one room captured twice at same tier (repeatability) | `Data/H2.MOV` | Second Hall video take exists; comparison run pending | PARTIAL — file captured, run not yet executed |
| Tape ground truth on everything; raw sensor data submitted | `ground_truth.json`, `Data/`, `CozmoData/` | ground_truth.json covers all 4 Data/ rooms (walls, ceiling, openings, damage); CozmoData/ has no tape truth | PARTIAL — CozmoData/ unmeasured (no Pro device, sample data only) |

---

## Part 3 — Head-to-head vs consumer app

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| 2 benchmark rooms, LiDAR tier output vs named consumer app on same rooms | — | — | FAIL — no Pro device; cannot run LiDAR tier on Data/ rooms to compare against a consumer app |
| One table: your error vs theirs, dimension by dimension | — | — | FAIL |
| Beat or tie on ≥70% of shared dimensions | — | — | FAIL |

---

## Part 4 — Fix loop

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Fix declaration: worst gate, failing number, root cause, predicted after | `Deliverables/fix_loop/declaration.md` | One-page declaration (draft exists; target pending final benchmark run) | PENDING |
| Shipped fix | `src/floorfathom/` (diff TBD) | Code change addressing the declared gate | PENDING |
| Before run (regenerable) | `out/fix_loop_before/plan.json` | Full plan.json from pre-fix run; command in declaration | PENDING |
| After run (regenerable) | `out/fix_loop_after/plan.json` | Full plan.json from post-fix run; same command | PENDING |
| Readable diff | git diff on changed files | `git diff <before-sha> <after-sha> -- src/` | PENDING |

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
| 5 | Benchmark report (gates at all 3 tiers, repeatability table, timing) | `Deliverables/benchmark_report.md` | PARTIAL — skeleton with known numbers; final numbers pending agent runs |
| 6 | Fix loop bundle | `Deliverables/fix_loop/` | PENDING |
| 7 | Technical report (max 6 pages) | `Deliverables/technical_report.md` | DRAFT — fix loop page placeholder |
| 8 | Raw benchmark data (sensor logs, ground truth) | `Data/`, `CozmoData/`, `ground_truth.json` | PARTIAL — ground_truth.json complete for Data/; CozmoData/ has no tape truth |

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
