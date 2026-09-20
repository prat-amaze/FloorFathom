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
- Flash: off. Do not use Night mode / long exposure.
- Turn on the room's lights and open the curtains before you start.

## How to walk — Photos tier (2–8 stills per room)

1. Stand in a corner of the room.
2. Take one photo facing each wall in turn, rotating in place (roughly 45° between
   shots), so consecutive photos overlap.
3. Include the floor and ceiling line in frame where possible — don't crop tight on
   eye-level only.
4. If the room has an alcove, closet opening, or a second doorway not visible from
   the first corner, walk to a second position and repeat step 2 for that area.
5. One photo folder per room. Do not mix rooms into one folder.

## How to walk — Video tier

1. Stand in a doorway or corner, phone upright (portrait) at chest height. Start recording.
2. Walk the full perimeter of the room once, pointing at the wall/floor/ceiling junction,
   moving slowly (a few seconds per wall).
3. Move your feet, not only your wrist: step sideways or forward while you pan. Turning
   on the spot gives no depth.
4. Never whip the phone quickly, for example through a doorway: blur breaks the
   reconstruction. Do not stare at a blank wall or a close-up of a door for more than
   2 seconds; keep furniture, door frames or corners in view.
5. Keep your hands, boxes and other objects out of the picture, and keep people and pets
   out of the room.
6. For a multi-room capture, keep recording continuously while walking through the
   connecting doorway/hallway into the next room — do not stop and restart between
   rooms.
7. Stop recording once you're back near your starting point. Keep each room under
   about 60 seconds.

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
5. Hand the pipeline one room's photo folder, or one video clip, per run.

## Device matrix

| Tier   | Device tested         | Notes                                                        |
|--------|------------------------|---------------------------------------------------------------|
| Photo  | iPhone 16 (non-Pro)    | 0.8x zoom, native Camera app                                   |
| Video  | iPhone 16 (non-Pro)    | 0.6x zoom, native Camera app. The clip records its own zoom and the pipeline reads it; other iPhone 15+ models are untested. Without a reference object the metric scale comes from a depth model and was off by -1%, +7% and +67% on three clips, so intervals are wide (at least 15%) and rooms are flagged |
| LiDAR  | *(none available)*     | No Pro-class device on hand; tier run against Cozmo-provided sample data instead of a live capture |

*(This matrix will be updated once a Pro-class device is available to capture and
verify the LiDAR tier live, and once accuracy numbers per tier are measured.)*
