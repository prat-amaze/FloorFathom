Use **Superpowers** throughout: inspect the existing video pipeline first, understand the current implementation and tests, produce a concrete implementation plan before coding, and validate each major change incrementally. Do not blindly rewrite working components.

## Goal

Improve the **video → metric floor-plan** pipeline.

The current high-level pipeline is:

`video → keyframes → pycolmap SfM → dense Depth Anything V2 cloud → metric scale → gravity → room estimator → RoomPlan → stitching → damage`

The current SfM, ruler-based scaling, gravity estimation, damage pipeline and output structure should be preserved unless there is a demonstrated reason to change them.

### Current failure

The weak stage is the **room estimator**.

Current results are poor:

* wall-length accuracy: 0/4 within ±3% on B1, B2 and H1
* opening accuracy: 0/6
* glossy floors produce sparse/fragmented floor points
* free-space becomes fragmented
* walls are consequently split into short pieces

There is already a prototype **camera-ray free-space carving** approach that produced the correct floor area on H1 but has not been merged because it touches the shared `layout.py` / `estimate.py`.

Inspect that prototype and evaluate it rather than assuming the current point-cloud-only approach is sufficient.

## Desired approach

Improve room estimation by combining multiple sources of geometric evidence:

1. **Dense point cloud**

   * Continue using the existing SfM + Depth Anything reconstruction.
   * Use it for wall/floor/ceiling evidence where reliable.

2. **Camera-ray / visibility evidence**

   * Use known camera poses and depth observations to infer free space even where the dense cloud is sparse.
   * A glossy/textureless floor should not cause the room polygon to fragment simply because few floor points were reconstructed.
   * Evaluate the existing ray-carving prototype against the current estimator.

3. **Wall reconstruction**

   * Detect and aggregate wall evidence across frames rather than treating fragmented point clusters as independent walls.
   * Fit robust wall planes/lines from all available observations.
   * Merge fragments belonging to the same physical wall.
   * Use architectural/Manhattan constraints where justified, without forcing genuinely irregular geometry.

4. **Floor-plan reconstruction**

   * Build the room boundary from the combined wall + free-space evidence.
   * Prefer a geometrically consistent closed polygon over a point-cloud convex hull or arbitrary closure.
   * Calculate floor area directly from the final polygon.
   * Produce accurate wall lengths and corners.

5. **Openings**

   * Improve doorway/opening detection using the reconstructed wall geometry, camera observations and visibility/depth evidence.
   * Do not infer openings merely from gaps in a sparse point cloud.
   * Preserve uncertainty when an opening is only weakly observed.

6. **Ceiling height**

   * Continue using the existing floor/ceiling reconstruction and metric scale.
   * Validate ceiling height independently from floor-plan accuracy.

## Important architecture

Do not create a completely separate room estimator just for one failure case.

Identify the reusable geometric abstraction between the video and other tiers where appropriate:

`geometry evidence → wall graph → room polygon → RoomPlan`

but keep video-specific inputs such as camera-ray visibility separate from LiDAR/photo-specific inputs.

The output must remain compatible with the existing `RoomPlan` and downstream damage stack.

## Validation

Use the existing synthetic tests to isolate stages where possible.

For real video data, evaluate against `ground_truth.json` using:

* wall-length error
* floor-area error
* corner-position error
* opening detection/position error
* ceiling-height error

The existing gates are the acceptance targets:

* wall lengths: ±3%
* ceiling height: ±1.5 cm
* opening measurements/positions according to the existing ground-truth evaluation

Do not tune specifically until a baseline is recorded.

Compare:

`current point-cloud estimator`
vs.
`ray/free-space estimator`
vs.
`combined estimator`

especially on H1's glossy floor.

## Performance

The current cold-run estimate is 15–20 minutes per clip against a 15-minute budget.

Do not sacrifice major accuracy gains merely to preserve the current runtime, but identify the expensive stages and avoid unnecessary recomputation. Preserve existing caching.

## Implementation order

1. Inspect current `layout.py`, `estimate.py`, SfM/dense reconstruction and the unmerged ray-carving prototype.
2. Establish baseline metrics on existing test rooms.
3. Integrate/evaluate ray-based free-space evidence.
4. Improve wall aggregation/fragment merging.
5. Reconstruct the room polygon from combined evidence.
6. Improve opening detection.
7. Validate measurements against ground truth.
8. Only then optimize runtime and clean up/refactor.

Do not modify damage detection unless the new RoomPlan geometry requires an interface-compatible change.

The primary objective is **accurate room geometry and measurements**, not visual similarity of the rendered floor plan.
