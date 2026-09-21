# Fix Loop Declaration — Video Tier

*Shipped fix. Before and after runs are regenerable (commands below); the readable diff is the three-file diff below.*

## 1. Worst-performing gate

**Gate:** Opening detection / position on the video tier (benchmark room `hall_kitchen`, H1.MOV).

**Failing number (before):** **1 of 6 openings detected.** The room has 6 tape-measured openings
(4 doors 71/81/81/107 cm, a balcony door 178 cm, a kitchen window 110 cm). The pre-fix run found
**one** (69 cm, the bathroom door) and missed the other five. Opening detection at 1/6 is the worst
single number in the video benchmark — worse than walls (0/4 within ±3%) because most openings were
not found *at all*, not merely mis-sized.

Reproduce the before number: `out/fix_loop_before/plan.json` → `rooms[0].openings` (length 1).

## 2. Root-cause hypothesis and evidence

**Hypothesis:** The floor height is estimated ~1.5 m too low because the glossy tile floor mirrors the
whole room *below* itself, and `planes.find_floor` locks onto that reflection. The doorway detector
(`video_openings.find_openings`) only counts through-wall rays that cross a wall between 0.1 m and 0.8 m
**above the floor** (a door reaches the floor, a window does not). With the floor 1.5 m too low, that
band sits in the reflection zone, well below every real door, so almost no through-rays are counted and
the doors are invisible.

**Evidence (all from H1's cached cloud, `out/h1/work`):**
- `find_floor` returns **y = −4.01 m**, but the true floor is a clean, strong spike at **y = −2.45 m**
  (support ≈ 13 000 points, 2 cm spread). The cloud extends down to −5.52 m: a mirror image of the room.
- `find_floor` searches for the floor within 1.2 m of the 0.5th height percentile. The reflection tail
  (≈ 5.4 % of points, from −5.52 to −2.65 m) drags that percentile to ≈ −4.5 m, so the 1.2 m window tops
  out near −3.3 m and the true floor at −2.45 m is **above the window and can never be chosen**.
- With the wrong floor the door-crossing band (floor + 0.1…0.8 m) is at −3.9…−3.2 m — empty air below
  the real doors. Before: 1 opening. After correcting the floor: **6 openings.**
- Same root cause is visible in two other gates: the wall-coverage band also samples the reflection
  zone (empty wall evidence → jagged outline) and the ceiling search is measured from the wrong floor.

**Rejected alternative:** the prior hypothesis in this file blamed outliers *above* the ceiling (balcony
glass, sky bleed) pulling the ceiling estimate up. The histogram shows the opposite — the biasing
population is *below* the floor (reflections), and it corrupts openings and walls, not just the ceiling.

## 3. Fix and predicted vs actual result

**Fix shipped:** `video_heights.floor_from_strongest_spike` finds the floor as the most-supported
histogram spike across the plausible height range (robust to the reflection tail), and `run_clip` drops
points below it before the estimator, ray carving and height refinement ever see the cloud.

**Predicted (before shipping):** openings recovered from 1/6 to ≥5/6 detected; floor height corrected to
≈ −2.45 m; at least one wall within the ±3 % gate. Position error on found openings improves but may not
all clear the strict ±2 cm gate, because the underlying cloud is still bowed and the capture path did not
sweep every wall.

**Actual (after):**

| Number | Before | After | Tape |
|---|---|---|---|
| Openings detected | **1 / 6** | **6 / 6** | 6 |
| Openings within 10 cm | 1 | 3 | — |
| Openings within 2 cm gate | 1 (69 vs 71) | 1 (108 vs 107) | — |
| Floor height | −4.01 m (reflection) | **−2.48 m (correct)** | ≈ −2.45 m |
| `kitchen_side_wall` | 608 cm (+15.6 %) | **527 cm (+0.2 %, passes ±3 %)** | 526 |
| Walls within ±3 % | 0 / 4 | 1 / 4 | 4 |
| Floor area | 23.7 m² | 23.2 m² | ≈ 24 |

**Verdict against the gate:** the opening gate moved from **fail (1/6 found)** to **all 6 found**, a
meaningful, correctly-diagnosed movement. It does **not** fully clear the ±2 cm position gate (still 1/6
within 2 cm): the three interior doors read 129–159 cm against 71–81 cm because the room outline they sit
on is still a sheared kite — the capture walked a diagonal path and never swept the left wall, so that
wall is unreconstructed and the door jambs on it are placed on free-space edges, not wall geometry. That
is a capture-coverage and depth-cloud-quality limit, not the floor bug this fix targeted.

## Regenerable before / after and diff

```bash
# BEFORE (video files at the commit before the floor fix)
git checkout 7055a33^ -- src/floorfathom/video_heights.py src/floorfathom/video_pipeline.py
uv run floorfathom plan Data/H1.MOV --out out/fix_loop_before --reference-length-cm 31.6

# AFTER (the floor fix)
git checkout 7055a33 -- src/floorfathom/video_heights.py src/floorfathom/video_pipeline.py
uv run floorfathom plan Data/H1.MOV --out out/fix_loop_after  --reference-length-cm 31.6
```

Openings: `plan.json` → `len(rooms[0].openings)` (before 1, after 6).
Floor: `work/alignment.json` → `floor_y` (before ≈ −4.0, after ≈ −2.48).

**Readable diff (the fix is 3 files, +106/-2 lines, inside commit `7055a33`, which also holds photo-tier changes):**
```bash
git diff 7055a33^ 7055a33 -- src/floorfathom/video_heights.py src/floorfathom/video_pipeline.py tests/test_video_heights.py
```

*Status: fix shipped (commit `7055a33`). The before and after numbers above were captured by the video session; `out/fix_loop_before` and `out/fix_loop_after` are not in the repository (`out/` is git-ignored), so regenerate them with the commands above (about 13 min per run). 43 video-tier tests pass, as reported by the video session.*
