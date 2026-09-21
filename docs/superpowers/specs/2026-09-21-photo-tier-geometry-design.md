# Photo-tier geometry: multi-view reconstruction, Manhattan walls, joint stitching

## Scope

In scope: the photo tier's per-room and whole-property geometry only —
`photo_pose.py`, `photo_scene.py`, `photo_layout.py`, `photo_reference.py`, and the
stitching path they feed (`stitch.py`). The goal is accurate wall geometry, corners, room
dimensions, floor area, ceiling height, openings, and calibrated uncertainty, plus a
correctly stitched whole-property plan from per-room photo folders.

Out of scope, unchanged: the LiDAR tier, the video tier's own pipeline (its *modules* are
reused, not modified), damage detection (`assess.py`, `damage.py`, `concealed.py`,
`scope.py`, `photo_damage.py`), the published schema (`schema.py` / `capture_plan.schema.json`),
and the existing uncertainty/reporting mechanism in `report.py`/`uncertainty.py` (reused as-is,
fed new noise sources — no new fields, no new diagnostics keys).

## Why: the capture protocol constraint

The current photo protocol (`capture_protocol.md`) mandates standing in one spot and
rotating in place — `photo_pose.py`'s own docstring says this is deliberate, because a
zero-baseline capture makes classical triangulation degenerate. Genuine multi-view
geometry (COLMAP/SfM, or any triangulation-based matcher) needs real camera translation to
not be degenerate. The video tier already requires this ("move your feet, not only your
wrist") and already runs real SfM (`sfm.py`, pycolmap) for exactly this reason.

**Decision:** change the photo capture protocol to require a short walking path per room
(a small arc/loop, matching the video protocol's existing language), and recapture
`Data/B1`, `B2`, `Hall` under the new protocol. `Data/ground_truth.json`'s tape
measurements remain the accuracy target.

## Architecture: module disposition

| Module | Disposition |
|---|---|
| `photo_pose.py` | **Replaced.** Rotation-only SIFT registration → real pycolmap SfM. Reuse the video tier's SfM machinery (`sfm.py`) where genuinely generic (the pycolmap wrapper, camera-model handling); keep photo-specific sparse/discrete-image handling separate from `sfm.py`'s temporal/keyframe assumptions rather than forcing one code path over both. SfM's recovered poses/geometry are the authoritative geometric scaffold. EXIF-derived per-photo focal length seeds pycolmap's per-image camera model when present and plausible; missing or implausible EXIF falls back to a documented default with a flag, the same pattern `sfm.py` already uses for video's rounded 35mm-equivalent metadata — never assumed correct. |
| `photo_scene.py` | **Replaced.** Cloud built as: SfM sparse cloud (arbitrary scale, primary geometry) → gravity alignment (reuse `world.py`, already generic over any `SfmResult`) → monocular-depth densification harmonized to SfM's own scale (reuse `points_video.py`'s fit-to-sparse-then-densify approach). The depth model is auxiliary support for coverage on blank walls, never the sole or authoritative geometric source. The dense cloud is validated against the SfM sparse points (consistency check) before being handed to wall fitting. |
| `photo_reference.py` | **Replaced.** Its own docstring states the constraint being removed: "the photos turn on the spot, so there is no parallax and a size cannot be triangulated." With real camera translation, swap the single-view wall-plane-ray method for `anchor.py`'s multi-frame ray-triangulation (already implemented for video, already produces a jackknife uncertainty estimate). The yellow-ruler mechanism stays the preferred absolute-scale source; scale uncertainty is propagated, not dropped. |
| `photo_layout.py` | **Kept, extended.** RANSAC vertical-plane wall fitting stays, now run on the richer SfM+dense cloud. New: an evidence-weighted Manhattan-angle optimization pass (below). Wall extents still come from actual point support; ray-cast closure chords are never treated as measured evidence. |
| `stitch.py` | **Kept, extended.** Sequential doorway-gluing stays as the initial solve and the fallback. New: a joint global refinement pass over all doorway/shared-wall constraints when enough independent constraints exist (below). One consistent metric 2D coordinate system throughout. |
| `estimate.py`, `layout.py` shared RANSAC/line-fit primitives | **Kept, reused as-is** — no duplication. |
| `report.py`, `assess.py`, `scope.py`, `concealed.py`, `schema.py` | **Untouched** unless implementation reveals a concrete incompatibility. They depend only on the finished `RoomPlan`, not on how the cloud/walls were built. |
| Video tier (`sfm.py`, `world.py`, `points_video.py`, `anchor.py`, `video_pipeline.py`) | **Untouched in behavior.** Photo adapts to what these modules already generically accept; nothing changes for video's own inputs/outputs. |

## Manhattan-world wall-angle optimization (`photo_layout.py`)

After the existing per-wall independent line fits (`fit_line`/`snap_polygon`, which already
compute a `support` score per wall from inlier coverage):

- For every corner where two walls' independently-fit angles land within a **configurable**
  tolerance of 0°/90°/180° (default ~12-15°), add a soft constraint pulling both walls'
  angles toward the exact multiple, weighted by both walls' support.
- Corners not already close to a Manhattan angle get no constraint — never touched, never
  forced. This is what preserves genuinely non-Manhattan rooms (Hall is documented as an
  irregular quadrilateral).
- Solve the joint angle graph (small weighted least-squares) per room.
- **Guardrail:** after solving, re-check each adjusted wall's agreement with its own
  supporting points (the same residual computation `fit_line` already does). If the
  optimization would materially worsen a wall's fit to its evidence, reject the adjustment
  for that wall and keep its independent fit instead.
- Only orientation is snapped; each wall's offset is refit along its corrected angle using
  its own original inlier points, so position stays evidence-based.
- `evidence: "closure"` walls are excluded entirely — no angle vote in, no snap out.
- Corners are rebuilt the same way as today (`intersect()`/`rebuild_polygon`) on the
  corrected lines.

## Stitching joint refinement (`stitch.py`)

- Run the existing sequential doorway-gluing first (unchanged) — gives an initial
  `(angle, shift)` per room and a `Placement` per room, anchored on one doorway pairing.
- Collect every valid doorway/shared-wall correspondence between now-placed rooms, not just
  the one each room was originally glued on.
- **Robustness:** treat correspondences as candidates, not commitments. Score each by
  geometric plausibility (consistent with the sequential solve, width/alignment agreement);
  discard outlier correspondences rather than forcing every one into the solve.
- If enough independent, non-outlier correspondences remain to over-determine the system,
  solve a small 2D pose-graph (nonlinear least squares, initialized from the sequential
  solve) with one room's pose fixed as the anchor, refining every room's pose jointly.
- If not enough constraints survive (the common case: a simple hallway + 2 rooms, one
  doorway each), skip refinement and keep the sequential solve as final — this is the
  explicit fallback, not an edge case to special-case away.
- No schema change: `Stitching.placements` carries the same fields, just better-refined
  numbers when refinement ran.

## Uncertainty propagation

Reuses the existing bootstrap/systematic-term mechanism in `uncertainty.py`/`report.py`
unchanged in mechanism and unchanged in schema — fed new noise sources instead of old ones:

- **Stochastic term:** SfM registration/reconstruction variability (resampling over which
  registered photos/tracks are used), replacing the old per-photo-depth leave-one-out
  resampling.
- **Systematic term:** the ruler's multi-frame triangulation uncertainty (`anchor.py`'s
  existing jackknife estimate), folded in the same way `points_video.py`/`anchor.py`
  already do for video.
- These two sources are kept distinct and are not double-counted against each other — the
  stochastic term captures reconstruction noise, the systematic term captures scale/reference
  error, combined the same way the existing systematic-term-plus-bootstrap pattern already
  combines terms elsewhere in `uncertainty.py`.
- Weak, inferred, or closure geometry keeps correspondingly wider intervals under this same
  mechanism (no special-casing needed: less/no point support already means less/no bootstrap
  agreement, which the existing `interval()` logic already widens for).
- No new diagnostics fields, no new schema fields. Any human-readable status (e.g. whether
  stitching refinement ran) rides in the existing free-text `diagnostics.notes` list if
  needed, not a new structured field.

## Validation

**Stage-isolated synthetic validation (available immediately, no recapture needed):**

- Extend `tests/synth.py`'s renderer with procedural per-wall texture (so SIFT has real
  keypoints) and camera paths that actually translate (a short walk/arc, not a fixed point),
  giving pycolmap a real baseline against exactly known ground-truth geometry.
- Point-cloud-level stages (wall RANSAC fitting, Manhattan-snap, stitching) are also tested
  directly against synthetic 3D geometry + known poses, bypassing SfM where the goal is
  isolating that specific stage — the same pattern `test_surfaces.py`/`test_assess.py`
  already use for the damage stack.
- Deterministic regression tests assert against known values for the four agreed metrics:
  wall-length error, corner-position error, floor-area error, ceiling-height error.

**Real-world validation pass (once `Data/B1`, `B2`, `Hall` are recaptured under the new
protocol):**

- A new eval script, matching the existing `scripts/eval_damage_photo.py`/`eval_video.py`
  pattern, runs the finished pipeline on the recaptured rooms and scores the same four
  metrics against `Data/ground_truth.json`'s tape measurements.
- Synthetic-test success is never treated as evidence of real-world accuracy. The real pass
  is what tunes the system and is reported separately, against the PS's own gates (±8%
  wall length, ±1.5cm ceiling height for the photo tier).

## Non-goals (explicitly rejected during design)

- MASt3R/DUSt3R or other baseline-tolerant dense-pointmap models — rejected once the
  protocol itself was changed to give a real baseline; classical SfM is sufficient and
  reuses proven, already-tuned code.
- LoFTR/LightGlue learned feature matching — kept SIFT via pycolmap; sparse photos are
  specifically shot ~45° apart to overlap, so matching is less starved than video's
  continuous stream. Revisit only if real validation shows SIFT is the actual bottleneck.
- A third `Wall.evidence` category ("geometrically inferred") beyond `wall_points`/
  `closure` — rejected; the existing two-value distinction plus interval width already
  encodes "how much to trust this," per the existing design philosophy of never returning
  a falsely confident number.
- New schema fields or new structured diagnostics for Manhattan-constraint counts,
  stitching-refinement status, or rejected correspondences — rejected; any such status is
  free text in `diagnostics.notes` if surfaced at all, not a schema change.
