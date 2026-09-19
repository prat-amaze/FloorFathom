# Capture Protocol (Route 2: Stock Capture)

**Tool:** iPhone native Camera app (no third-party capture app, no dev build required).
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
- For video, switch to Video mode and set zoom to **0.6x** before recording.
- Flash: off. Do not use Night mode / long exposure.

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

1. Stand in a doorway or corner. Start recording.
2. Walk the full perimeter of the room once, keeping the phone at chest height,
   pointing at the wall/floor/ceiling junction, moving slowly (a few seconds per wall).
3. For a multi-room capture, keep recording continuously while walking through the
   connecting doorway/hallway into the next room — do not stop and restart between
   rooms.
4. Stop recording once you're back near your starting point.

## LiDAR tier

No LiDAR-capable device (iPhone Pro/Pro Max) was available for this capture round.
This tier is exercised against **Cozmo-provided sample data** instead of a live
capture. 

## Handoff to the pipeline

1. Copy all photos for a room into a folder named after the room
   (e.g. `images/Hall/`, `images/b1/`). Do not rename individual files.
2. Copy each room's video as a single `.MOV` file named after the room
   (e.g. `Hall.MOV`). If a room was captured twice (repeatability), suffix the
   second file with `_2` (e.g. `Hall_2.MOV`).
3. Upload Everything to a drive(I used Google drive) and download in the Laptop.
3. Hand the top-level folder (containing the per-room image folders and the `.MOV`
   files) to the pipeline. No renaming, no re-encoding, no cropping.

## Device matrix

| Tier   | Device tested         | Notes                                                        |
|--------|------------------------|---------------------------------------------------------------|
| Photo  | iPhone 16 (non-Pro)    | 0.8x zoom, native Camera app                                   |
| Video  | iPhone 16 (non-Pro)    | 0.6x zoom, native Camera app                                   |
| LiDAR  | *(none available)*     | No Pro-class device on hand; tier run against Cozmo-provided sample data instead of a live capture |

*(This matrix will be updated once a Pro-class device is available to capture and
verify the LiDAR tier live, and once accuracy numbers per tier are measured.)*
