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
  everything after 8.84 s.
- **Motion blur:** in `B1` the chosen keyframes range from sharpness 3.3 to a median of
  54, so some 0.15 s windows contain only blurry frames.
- **No loop closure:** matching each keyframe with its 12 neighbours registered 77% of
  `B2` where matching all pairs registered 91%, because the links from the end of the walk
  back to its start are gone. All pairs is not an option for long clips (530 s for the
  22 s `B2`, and `Full.MOV` is 95 s).
- **Blank ceilings:** in `B1` and `B2` the sparse points pile into a floor layer but no
  ceiling layer shows among the tallest height bins, so ceiling height needs the dense
  depth stage and is `null` without it.
- **Scale is unknown:** SfM output has the right shape in arbitrary units. Until the
  metric-scale stage exists, no length from the video tier is a length in metres.
- **Gravity rests on the phone being held upright:** the layers of floor and ceiling fix
  the tilt but not the sign, so an upside-down capture would be aligned with the floor on
  top and nothing would notice. A phone rolled the whole capture on a single heading is
  flagged (`gravity_uncertain`).
- **Clip names do not follow the protocol:** the files are `B1`, `B2`, `BR`, `H1`, `H2`
  (two Hall takes) and `Full`, but `capture_protocol.md` asks for `Hall.MOV` and
  `Hall_2.MOV`. A clip cannot be tied to a room by name without renaming or a table.

Expected from how the method works, not yet seen in our data:

- **Lens model tied to one device and zoom:** the focal length is held at 0.6234 of the
  long image side (about 798 px at 720 px wide) with zero distortion. It is right for the
  iPhone 16 ultra-wide at 0.6x only; another device or zoom gives a wrong shape without
  any error. `ground_truth.json` also notes the 0.6x lens distortion is not corrected.
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
- **Depth network (not built yet):** it can be biased on rooms it has not seen, can
  invent depth on blank walls, mirrors and glass, and its scale error goes straight into
  every length. The expected 3-8% is unverified; the check is the `B2` ceiling (2.79 m).
- **Run-to-run differences:** COLMAP mapping is multithreaded, so identical input may not
  give an identical model, which the README's same-input-same-JSON promise would need.
