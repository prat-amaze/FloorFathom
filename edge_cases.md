# Edge Cases

Edge cases present in the capture data that the pipeline must handle.

## 2026-09-19

- **Mirrors:** full-length mirror in Bedroom 2 (`images/b2`, shows a full-body
  reflection) and a small wall mirror over the wash basin in Hall. Reflected space
  must not be reconstructed as real geometry.

- **Reflective floor:** glossy white floor tiles in Hall. Reflections must not be
  read as geometry or damage.

- **Open-plan Hall:** kitchen, living/dining and entry share one space and one photo
  folder. Must produce one room with the zones inside it, not separate rooms.

- **Non-rectangular rooms:** Hall is an irregular quadrilateral and the bathroom is
  L-shaped. Right angles must not be assumed.

- **Hall damage (in the `Hall` set):** a branching hairline crack on the kitchen
  backsplash below the chimney hood (12 × 22 cm), and a staged leakage mark
  (tea-stained paper with drip trails) on the wall beside the bathroom door
  (21 × 25 cm).

- **Bathroom damage:** real ceiling crack. Must be detected and kept separate from
  the Hall damage.

- **Low light:** Hall photos and videos were re-captured in the evening under
  artificial light, with a dark balcony behind the glass door.

## LiDAR tier (Cozmo sample data, 2026-09-20)

Not handled yet by `points.py` (depth to point cloud), which filters only by confidence
(at least 2) and range (0.3 to 4.0 m):

- **Pose drift:** poses are used as-is, with no loop closure or other correction. Small
  for one room, but it accumulates along a long multi-room walk and will bend or double
  walls when rooms are stitched.
- **Stray depth pixels:** flying pixels at depth edges (door frames, furniture corners)
  are not removed and add points in mid-air.
- **Mirrors and glass:** depth on a mirror, window or glass door is wrong (the sensor
  sees through or reflects), and those points enter the cloud unflagged as real geometry.
- **Moving objects:** a person or pet crossing the view leaves ghost points that stay in
  the cloud.

## Video tier (iPhone 16 `.MOV`, 2026-09-20)

The clips carry no IMU, no depth and no gravity, only lens info, a 35 mm-equivalent focal
length and lux. Everything else is recovered from the pictures (SfM in `sfm.py`, gravity in
`world.py`). Numbers come from `scripts/spike_video_sfm.py` and the video modules.

Seen in our clips:

- **Capture breaks (fast pan, blur, blank surface):** SfM links a frame to the next only
  through shared features. In `B1` keyframes 64-72 are a fast pan through a doorway
  (blurred), 74-83 are a plain door leaf and then a blank wall; verified matches between
  consecutive keyframes fall from 450-880 to 0-40 and the reconstruction splits. `BR`
  breaks around keyframes 26-30 (largest piece 46%). Denser keyframes and a lower feature
  threshold did not help, because the frames themselves have nothing to match. Walls seen
  only after a break get no pose and no numbers (`sfm_capture_break`); in `B1` that is
  everything after 8.84 s. `B2` breaks at 8.8-11.5 s (a pan from the curtain to a bare
  corner) and 19.8-21.7 s (a blank wall at the end): the frames without a pose hold 230-850
  SIFT points against 2000-4000 on registered ones. It is not the matching window: COLMAP's
  sequential matching already pairs frames 16 and 32 apart. Raising local contrast (CLAHE)
  before feature detection was tried and made it worse: registered keyframes fell from
  126 of 164 to 65 (77% to 40%), reprojection error rose from 0.67 to 0.99 px and SfM took
  777 s, because noise and floor gloss were amplified along with the wall shading. Reverted.
  What is left is bridging with depth-model geometry (not built) and the protocol rule
  never to pan across a bare wall.
- **Motion blur:** in `B1` the chosen keyframes range from sharpness 3.3 to a median of
  54, so some 0.15 s windows contain only blurry frames.
- **No loop closure:** matching each keyframe with its 12 neighbours registered 77% of
  `B2` where matching all pairs registered 91%, because the links from the end of the walk
  back to its start are gone. All pairs is not an option for long clips (530 s for the
  22 s `B2`, and `Full.MOV` is 95 s).
- **Blank ceilings:** in `B1` and `B2` the sparse points pile into a floor layer but no
  ceiling layer shows among the tallest height bins, so ceiling height needs the dense
  depth stage and is `null` without it.
- **Scale comes only from a depth model, and it is off by different amounts per scene:**
  SfM output has the right shape in arbitrary units. The Depth Anything V2 metric-indoor
  model, fitted to SfM depth, gave a scale error of -1% (`B2`), +7% (`B1`) and +67% (a
  practice clip, `IMG_4765`, outside the benchmark set), with 10-17% spread between frames of one clip, and the
  result also depends on the input size given to the model. Every video room is flagged
  `scale_from_depth_model_only` and every interval carries at least 15% relative scale
  uncertainty, so intervals are wide (`B2`: area +-40%, ceiling +-50%) and the video
  gate of +-3% on wall lengths cannot be met without a reference object of known length.
- **A reference object only works if it stays put:** the scale anchor (a strip of known
  length, `anchor.py`) needs the camera to move sideways around an object fixed in the
  room. An object carried in a hand or a box moves with the camera and gives no parallax,
  and a turn on the spot gives none either. Baselines under 5 degrees are flagged
  (`anchor_weak_baseline`); on a weak-baseline test the resampling error alone said 12.8%
  where the real error was 29%, so a pixel-noise term was added. The capture protocol now
  asks for a yellow 30 cm ruler on a wooden surface at the start of each clip; a clear or
  white ruler is not found, and a light door frame can be mistaken for a white strip.
- **Registration varies a lot by clip:** keyframes with a pose in the kept model were
  `B2` 77%, `B1` 66%, `H1` 45%, `BR` 46% and `IMG_4765` 100%. Below 60% the pipeline
  gives no rooms and a reason (`sfm_registered_too_few_frames`), so two of the five
  real room clips give no numbers at all today.
- **The room estimator was tuned on LiDAR, not depth-model clouds:** on `B2` it returned
  13 wall pieces (true room: 4 walls, 3.6 x 3.0 m), a second room of 2.6 m2 with no
  ceiling, area 11.79 m2 against 10.88 (+8%) and ceiling 3.33 m against 2.79 (+19%). Depth
  models also bend flat walls slightly, which the plane thresholds treat as noise or as
  separate walls.
- **Gravity from geometry is rough on real clips:** the height-layering cue was about 7
  degrees off on real footage and the floor refinement on `B2` found the floor but no
  ceiling. A cloud of walls only gave a spurious 17 degree correction with loose
  thresholds, so refinement needs both a floor-like plane and a correction under 15
  degrees (`gravity_no_horizontal_plane`, `gravity_correction_too_large`).
- **Time and memory:** one clip takes about 9 minutes on this 7.3 GB machine (`B2` 538 s
  end to end), and a run next to a browser and the full test suite was killed for lack of
  memory. The SfM and depth results are cached under `work/` so a killed run resumes.
- **Gravity rests on the phone being held upright:** the layers of floor and ceiling fix
  the tilt but not the sign, so an upside-down capture would be aligned with the floor on
  top and nothing would notice. A phone rolled the whole capture on a single heading is
  flagged (`gravity_uncertain`).
- **Clip names do not follow the protocol:** the files are `B1`, `B2`, `BR`, `H1`, `H2`
  (two Hall takes) and `Full`, but `capture_protocol.md` asks for `Hall.MOV` and
  `Hall_2.MOV`. A clip cannot be tied to a room by name without renaming or a table.
- **Zoom differs between clips of one phone:** the 35 mm-equivalent focal length written
  into the clips reads 16 (`B1`, `B2`, `BR`), 15 (`H2`) and 14 (`H1`, `Full`), all on the
  same lens. The focal length found by SfM follows it: about 798 px at 16 mm and 730 px
  at 14 mm (720 px wide). A single fixed value was wrong by about 9% on `H1` and raised its
  reprojection error by about 15% (0.72 to 0.84 px). The metadata is a rounded whole
  number and leaves out the stabilisation crop, so it is used as a starting value that SfM
  refines (`focal_far_from_metadata` if the result moves more than about 15-18%). The
  protocol's "0.6x" zoom may not exist on other iPhone models, and the defense uses the
  tester's own phone.

Expected from how the method works, not yet seen in our data:

- **Lens distortion:** it is fixed at zero (refined values came out at 0.00 on the iPhone
  16), but `ground_truth.json` notes the 0.6x lens distortion is not corrected, and
  another device may not behave the same.
- **Rotation metadata:** the `.MOV` stores a 90 degree rotation. Frames come out upright
  only because OpenCV applies it; a decoder that ignores it would give sideways frames
  and a wrong principal point.
- **Mirrors (B2 full-length, Hall wash basin):** reflected features triangulate as
  points behind the mirror, a phantom room. A depth network is likely to report the
  depth of the reflected room as well.
- **Glossy Hall floor:** reflections shift with the viewpoint, so they match badly or
  triangulate below the floor and make a second layer that blurs the floor plane.
- **Glass and the balcony door (Hall):** transparent and reflective; at night the dark
  glass reflects the room, and depth on it is wrong.
- **Low light (Hall, evening):** longer exposures mean more blur and noise and fewer
  features, so more capture breaks than the daytime clips.
- **Look-alike places:** the identical door leaves of B1 and B2, uniform paint and floor
  tiles can be matched to the wrong place, most likely across rooms in `Full.MOV`.
- **Moving things:** a person, a pet or a swinging door leaf breaks the assumption that
  the scene is rigid and leaves outlier points (in `B1` keyframes 74-78 a door leaf is
  close to the lens; not checked whether it was moving).
- **Long-walk drift:** on `Full.MOV` scale and heading errors accumulate with no loop
  closure, and a room visited twice will not line up with itself.
- **Depth network on blank walls, mirrors and glass:** it can invent depth there, and its
  scale error goes straight into every length (measured above, not the 3-8% first hoped).
- **Run-to-run differences:** COLMAP mapping is multithreaded, so identical input may not
  give an identical model, which the README's same-input-same-JSON promise would need.
  The `H1` against `H2` repeatability check could not be scored: `H1` registers 45% of its
  keyframes and gives no room, and `H2` gives one room (31.85 m2 against a tape-measured
  hall of about 5.3 x 4.7 m, ceiling 4.68 m against 2.79 m, 3 of 4 tape walls collapsed to
  under 2 m, 0 of 5 doors within 2 cm).
- **Windows are not told from doors:** the schema has one opening kind, `doorway`, so a
  window such as `B2`'s (110 x 240 cm) is either missed or reported as a doorway.
- **Several clips in one folder:** the video tier takes one clip per run, and stitching
  several video rooms into one plan is not built, though the LiDAR tier's stitching exists.
- **Zoom changed while recording, or a clip sent through a messenger:** the focal length is
  fixed for a whole clip, so a zoom change mid-clip breaks SfM. A clip re-encoded by
  WhatsApp loses the lens metadata, the pipeline then starts from a default focal length
  (`focal_prior_default`) and has lower resolution to work with.
- **Hands, boxes and people in frame:** points on things that move with the camera or move
  in the room are outliers in SfM and in the depth fit.
