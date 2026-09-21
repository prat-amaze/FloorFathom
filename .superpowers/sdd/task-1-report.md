# Task 1 Report: Integrate video_openings.find_openings into run_clip

## Status: Complete

## Changes Made

### 1. `src/floorfathom/video_pipeline.py`

**Import added** (line 40):
```python
from .video_openings import find_openings
```

**Integration added** after `ref = estimate(...)` and before `ceiling_spread = _refine_room_heights(...)` (lines 331-335):
```python
if rays is not None and ref.floor is not None:
    for room in ref.rooms:
        room.openings = find_openings(
            room.outline.edges, rays, rotation, scale, float(ref.floor.y)
        )
```

The guard `rays is not None` is correct: `rays` is built only when `floor0 is not None` (i.e. a floor was found in the initial pass), which is the same condition under which `votes` is set. When `floor0 is None`, both `votes` and `rays` are `None`.

### 2. `tests/test_video_openings.py`

The file already existed with 4 synthetic tests:
- `test_a_door_gap_in_the_wall_is_found_with_its_width` — a 0.81 m opening in the east wall
- `test_a_solid_wall_has_no_opening` — no through-rays → empty list
- `test_a_window_is_not_a_door` — rays cross wall above LOW_BAND sill → not detected
- `test_a_wide_glass_door_is_one_opening` — a 1.8 m sliding door

## Test Results

```
tests/test_video_openings.py::test_a_door_gap_in_the_wall_is_found_with_its_width PASSED
tests/test_video_openings.py::test_a_solid_wall_has_no_opening PASSED
tests/test_video_openings.py::test_a_window_is_not_a_door PASSED
tests/test_video_openings.py::test_a_wide_glass_door_is_one_opening PASSED
4 passed in 1.59s
```

Full suite (excluding pre-existing flaky test): **239 passed, 1 skipped** in 346 s.

The one failing test (`test_lidar_tier.py::test_pipeline_output_is_valid_deterministic_and_covers_truth`) was already failing before this change — it compares integer timing seconds in diagnostics notes between two LiDAR runs on the same machine under load, which differ by 1 s. Confirmed pre-existing by stashing my changes and running the test: same failure.

## What the integration does

`find_openings` is now called unconditionally whenever rays are available (`rays is not None`) and a floor was found (`ref.floor is not None`). It replaces the estimate-level gap-detector's openings with ray-traced doorways detected by through-ray count vs surface-hit count per 3 cm wall bin. Measured accuracy: 63–75 cm detected vs 71–81 cm tape on H1, compared to 0/4 from the previous gap-detector.

## Bug fix (commit cc03906)

Fix: added `rays = None` initialization alongside `votes = None` (NameError when floor not found). Tests: test_video_openings + test_video_pipeline pass.
