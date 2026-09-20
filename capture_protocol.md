# Capture Protocol (Route 2: Stock Capture)

**Tool:** iPhone native Camera app for the Photo and Video tiers (no third-party app, no dev build). The LiDAR tier needs a LiDAR logging app, see below.
**Devices used so far:** iPhone 16 (non-Pro — no LiDAR scanner on this hardware).

This page is followed literally at the defense. If a step below is ambiguous, that
ambiguity is a bug in this document, not in the tester's judgment.

## What to install

Nothing. Use the Camera app that ships with iOS. Do not install a third-party
scanning app for the Photo or Video tiers.

## Settings before you start

- Open Camera app → Photo mode.
- Hold phone in Portrait mode
- Set zoom to **0.8x** (ultra-wide) before taking room stills. Do not use 1x or 2x.
- For video, switch to standard Video mode (not Cinematic, not Action) and set zoom to
  **0.6x** (the ultra-wide lens) before recording. Do not change the zoom while recording.
  Set Settings → Camera → Record Video to **1080p HD at 30 fps** (all our clips are this).
- Flash: off. Do not use Night mode / long exposure.
- Turn on the room's lights and open the curtains before you start.

## How to walk — Photos tier (2–8 stills per room)

Before the first room: take the yellow ruler used for video (yellow body 31.6 x 4 cm). Measure the yellow part end to end with a tape to 1 mm, write the value down, and give it to the command with `--reference-length-cm` (as the video section says).

For each room:

1. Stick the ruler flat on a wall, upright, with its centre at about phone height (1.2-1.5 m), on a wood-coloured or dark surface such as a door or a wardrobe. Not on a mirror, window, glossy tile or glass door. Nothing else yellow next to it.
2. Stand facing that wall squarely, 1.0-2.5 m from it, phone upright (portrait) and level: turn no more than 15 degrees to either side of straight at the wall, and tilt the phone up or down no more than 15 degrees. This spot is where you take all of the room's photos. Keep the whole ruler in the middle third of the picture, sharp, with clear space around it and no hand, shadow or glare on it. Take the room's first photo.
3. From the same spot, take one photo facing each wall in turn, rotating in place (roughly 45 degrees between shots), so consecutive photos overlap. Leave the ruler in place until the last photo.
4. Include the floor and ceiling line in frame where possible; do not crop tight on eye level only.
5. If the room has an alcove, closet opening, or a second doorway not visible from your spot, walk to a second position and repeat step 3 for that area.
6. One photo folder per room, one ruler per folder. Do not mix rooms into one folder.

## How to walk — Video tier

Record one room per clip, each clip with its own ruler. A video gives the room's shape but
not its size, so every clip needs one ruler of known length that stays still in the room.

1. Take a straight plastic ruler with a **bright yellow body**. Measure the yellow part with
   a tape, from its top edge to its bottom edge, to 1 mm (ours: 31.6 cm), and write the
   number down: it goes into the run command (see Handoff).
2. Tape the ruler flat and upright (vertical) at about chest height, on a wood-coloured or
   dark surface such as a wooden door leaf or a wardrobe, in the room you are about to record.
   Nothing else yellow next to it. Never held in your hand, never on something that moves.
   Leave it there for the whole clip.
3. Open every door of the room, so each doorway is seen.
4. Stand 1.5–2.5 m from the ruler, phone upright (portrait) at chest height, the whole ruler
   in the middle of the picture with clear space on all four sides of it. Start recording.
5. **The ruler move (about 8 seconds):** walk a small arc around the ruler: about 1 m to
   your left, then about 1 m back to your right, feet moving. Turn the phone as you go so the
   ruler stays in the middle of the picture, fully in view. Do not keep the phone square to
   the wall, and do not turn on the spot: the ruler has to be seen from different places.
   Then go straight into step 6 without stopping.
6. Walk the full perimeter of the room once, pointing at the wall/floor/ceiling junction,
   moving slowly (a few seconds per wall). Point the phone at the ceiling corners at least
   once: a ceiling that is never seen gives no ceiling height.
7. Move your feet, not only your wrist: step sideways or forward while you pan. Turning
   on the spot gives no depth.
8. Never whip the phone quickly, for example through a doorway: blur breaks the
   reconstruction. Do not stare at a blank wall or a close-up of a door for more than
   2 seconds; keep furniture, door frames or corners in view.
9. Keep your hands, boxes and other objects (except the ruler) out of the picture, and
   keep people and pets out of the room.
10. Avoid holding the phone on mirrors, glass doors and bright windows: reflections and
    see-through glass give wrong geometry. Glance past them, do not linger.
11. Stop recording once you're back near your starting point. Keep each room under
    about 60 seconds.

## Ground truth and repeat takes (benchmark rooms)

- Before you leave a room, measure with a tape and write down: each wall's length at floor
  level, the ceiling height, each door's and window's width, and the ruler's yellow
  length (to 1 mm).
- To test repeatability, record the room a second time on the same route with the ruler in
  the same place, as a fresh recording. Name the second file with `_2` (see Handoff).

## LiDAR tier

Needs an iPhone Pro or Pro Max (LiDAR scanner).

**Install:** Stray Scanner (free, App Store). Its output folder (`rgb.mp4`, `depth/`,
`confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`) is the layout the pipeline
reads, as in the Cozmo sample scans. No Pro device was available to us, so this has
been checked against the sample scans only, not recorded live.

**How to walk** (one continuous recording for the whole property):

1. Start in the largest room and press record. Hold the phone upright, steady, at chest height.
2. Walk slowly (about one step per second) along the walls of each room, then pan the phone up
   to the ceiling and down to the floor once per room. A ceiling that is never seen gives no
   ceiling height.
3. Stay 0.5 to 3.5 m from walls; the pipeline ignores depth closer than 0.3 m or beyond 4 m.
4. Walk through every doorway slowly, in and out, with every door open. A doorway the
   camera never passed through is not found, and a closed door hides it.
5. End at the spot where you started, so the walk closes on itself.
6. Keep each recording under about 4 minutes (the sample scans are 0.6 to 3.6 minutes).

**Avoid:** mirrors, glass doors and large windows in view (depth is wrong on them), people or
pets moving through the frame, shiny wet floors, and fast turns.

**Hand-off:** copy the whole scan folder unchanged (AirDrop or a cable, not a messenger) and run
`uv run floorfathom plan <scan_folder> --out out/<name>`. One folder is one property.

## Handoff to the pipeline

1. Copy all photos for a room into a folder named after the room
   (e.g. `images/Hall/`, `images/b1/`). Do not rename individual files.
2. Copy each room's video as a single `.MOV` file. Name it after the room (e.g.
   `Hall.MOV`); the file name becomes the room name in the output. If a room was captured
   twice (repeatability), suffix the second file with `_2` (e.g. `Hall_2.MOV`).
3. Send the **original** file: AirDrop, a USB cable, or a drive's original-quality
   download (I used Google Drive). Never send it through WhatsApp or another messenger:
   they shrink the video and strip the lens information the pipeline reads.
4. Do not re-encode, crop or trim the video.
5. Hand the pipeline the photo folder that holds all the room folders, or one video clip, per
   run. Give the yellow length you measured. Photos: `uv run floorfathom plan <capture> --tier photo
   --out out/<name> --reference-length-cm <length in cm>` (the rooms are stitched into one plan).
   Video: `uv run floorfathom plan <clip> --out out/<name> --reference-length-cm <length in cm>`.

## Device matrix

| Tier   | Device tested         | Notes                                                        |
|--------|------------------------|---------------------------------------------------------------|
| Photo  | iPhone 16 (non-Pro)    | 0.8x zoom, native Camera app. The metric scale comes from the yellow reference ruler in each room (accuracy on real photos not measured yet). Without it, a depth model gives the scale and read ceilings +6%, +46% and +63% too tall on three rooms, so intervals are wide and rooms are flagged |
| Video  | iPhone 16 (non-Pro)    | 0.6x zoom, native Camera app. The clip records its own zoom and the pipeline reads it; other iPhone 15+ models are untested. The metric scale comes from the yellow reference ruler (31.6 cm body) (accuracy on real clips not measured yet). Without it, a depth model gives the scale and was off by -1%, +7% and +67% on three clips, so intervals are wide (at least 15%) and rooms are flagged |
| LiDAR  | *(none available)*     | No Pro-class device on hand; tier run against Cozmo-provided sample data instead of a live capture |

*(This matrix will be updated once a Pro-class device is available to capture and
verify the LiDAR tier live, and once accuracy numbers per tier are measured.)*
