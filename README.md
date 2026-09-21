# FloorFathom

Handheld iPhone captures to a dimensioned, stitched floor plan with damage regions and scope items, one command
per capture. Built for the Cozmo AI case study (`Applied AI.pdf`). Three input tiers (LiDAR, video, photos) produce
the same JSON (`schema/capture_plan.schema.json`).

| Tier | State |
|---|---|
| LiDAR (Stray Scanner folder: depth, poses, intrinsics) | Runs end to end: per-room plans, stitched plan, drift correction, damage. No tape truth exists for `CozmoData/`, so accuracy is unmeasured. |
| Video (one `.MOV` per room, yellow ruler taped up) | Runs end to end. Misses the +-3% wall and the opening gates on the real clips. |
| Photos (HEIC stills, one folder per room) | Runs end to end. Areas are null in all three rooms of `Data/`; the +-8% wall gate is not met. |

Measured results and their caveats: `Deliverables/benchmark_report.md` and `docs/verification_notes.md`.
Regenerating them from raw inputs: the separate reproduction bundle (not in this repo).

## Quick start: clean machine to first plan

Needs `git`, [`uv`](https://docs.astral.sh/uv/) and about 2.5 GB of free disk. Python 3.11 is fetched by `uv` if the
machine lacks it. Nothing else is installed by hand: video and HEIC decoding come in the Python wheels (no ffmpeg).
On a bare Windows machine the Microsoft Visual C++ 2015-2022 x64 runtime may be needed for torch (not tested without it).

```
git clone https://github.com/prat-amaze/FloorFathom.git FloorFathom && cd FloorFathom
uv sync --locked                   # 2.6 min cold, includes CPU torch from download.pytorch.org
uv run floorfathom fetch-models    # 38 s, 99 MB depth model from Hugging Face; the only step needing the network after uv sync
uv run floorfathom plan <capture> --out out/<name>
```

Run `fetch-models` before the first photo or video capture. Without it those tiers fail late (photo after SfM, video
after its roughly 14 minute SfM stage). After `fetch-models` everything runs offline.

Measured on one Windows 11 laptop (12 cores, 7.3 GB RAM, fast network, Python 3.11 and the VC++ runtime already present):

| Step | Time |
|---|---|
| `uv sync`, cold uv cache | 154 s |
| `fetch-models` | 38 s |
| Ready to run `plan` | about 3.5 min (Python download, if absent, adds 15 s; a slow link stretches the two downloads) |
| First LiDAR plan, cold (`single_room` 2.0 min, `single_scan_floor_only` 4.2, `single_scan_with_ceiling` 7.2) | 2-7 min |
| First photo plan (one room about 1-1.5 min, three rooms 4 min cold) | 1-4 min |
| First video plan, one clip, cold | **13.6-19.4 min**, so video does not fit the 15 minute budget on this machine |

So install plus a first plan lands under 15 minutes for the LiDAR and photo tiers. For video, install is ready in
about 4 minutes and the cold SfM stage takes 14 to 20 minutes per clip (clips run one after another); a rerun into the
same `--out` reuses `work/` and takes about 100 s. Every `uv run` spends about 3.5 s importing before it starts.

## One command per capture

Before you capture: follow `capture_protocol.md` literally. For photos and video, tape-measure the yellow ruler's yellow
body and pass it as `--reference-length-cm` (ours is 31.6). Copy the original files off the phone (AirDrop, cable),
never through a messenger, and do not rename, crop or trim them.

### LiDAR: one continuous walk of the whole property

```
uv run floorfathom plan <scan_folder> --out out/<name>
```

`<scan_folder>` is a Stray Scanner export, unchanged: `depth/<id>.png` (uint16, mm), `confidence/<id>.png`,
`odometry.csv` (columns `timestamp, frame, x, y, z, qx, qy, qz, qw`, optionally `fx, fy, cx, cy`), `camera_matrix.csv`
(read only when `odometry.csv` has no `fx`) and `rgb.mp4` (damage uses it). LiDAR depth is metric, so no ruler is needed;
`--reference-length-cm` is accepted and ignored (a note says so in `plan.json`). One scan folder per command (`CozmoData/` itself is the parent of three scans).
Live capture on a Pro iPhone is untested; the Cozmo sample scans in `CozmoData/` are the LiDAR data we ran.

### Video: one clip is one room, a folder of clips is one stitched property

```
uv run floorfathom plan path/to/Hall.MOV --out out/hall --reference-length-cm 31.6         # one room
uv run floorfathom plan path/to/folder_of_clips --out out/flat --reference-length-cm 31.6  # whole property
```

Each clip starts with the ruler taped up and in view for the first 15 s (the "ruler move" in `capture_protocol.md`).
The room is named after the clip's file stem. The folder must hold only that property's `.MOV` or `.mp4` files, at top
level. A room with no doorway that fits stays in its own frame and is listed in `stitching.unplaced`.

### Photos: one folder per room, all rooms stitched

```
uv run floorfathom plan <root> --tier photo --out out/<name> --reference-length-cm 31.6
```

`<root>/<room>/*.HEIC` (or `<root>/images/<room>/*.HEIC`; `.heic .heif .jpg .jpeg .png` are read). The subfolder name is
the room name. A single room folder works the same way: `uv run floorfathom plan <room_folder> --tier photo ...`.
Always write `--tier photo`; automatic detection fails on a flat folder of stills.

### How the tier is chosen

Automatic, in this order: a `.mov`/`.mp4` path is video; a folder with `depth/` and `odometry.csv` is LiDAR; a folder with
a top-level `.MOV`/`.mp4` is video; a folder with an `images/` folder or any subfolder is photo; anything else is an
error. Where that guess is wrong pass `--tier lidar|video|photo`. Known traps: `Data/` mixes `.MOV` files and room
folders, so it needs `--tier photo` (or copy the wanted clips into a fresh folder for video), and a LiDAR folder without
`odometry.csv` is silently routed to photo or video.

### Options

| Flag | Meaning |
|---|---|
| `--out DIR` | output folder (default `out`) |
| `--tier {lidar,video,photo}` | override tier detection |
| `--reference-length-cm CM` | photo and video: tape-measured length of the ruler's yellow body (default 31.6); ignored by LiDAR |
| `--seed N` | random seed (default 0); LiDAR repeats exactly, photo when its depth cache is present, video does not (see below) |
| `--bootstrap N` | replicates for the intervals (default 20); LiDAR and video only, photo uses leave-one-photo-out |
| `--no-debug` | skip debug output; LiDAR and photo only (video always writes its damage images) |

## Reading the result

Every run prints a summary and `wrote <out>/plan.json and <out>/plan.png`.

- `plan.png`: the top-down plan with wall lengths, area, ceiling height, doorways and damage marked. Open this first.
- `plan.json`: the contract. Every measurement is `{value, lo, hi, unit, method}`, a 95% interval; a quantity that was not
  measured is `null` with a note, never a guess. Check each room's `flags` and `diagnostics.notes`, which say what could
  not be judged. The stitched plan adds `stitching` (adjacency, overlaps, footprint, placed and unplaced rooms).
- `rooms/<room>.json`: each room's own plan. `debug/damage/<surface>.png`: each wall, floor and ceiling unrolled, damage
  outlined. LiDAR also writes `debug/points.ply`, `topdown.png`, `height_hist.png`.
- `work/` (photo and video): caches (depth maps, SfM, dense cloud). A rerun into the same `--out` reuses them.

Validate a `plan.json` against the schema (the schema is generated from the same model, `uv run floorfathom schema`):

```
uv run python -c "import sys,pathlib; from floorfathom.schema import CapturePlan; p=CapturePlan.model_validate_json(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); print('valid', p.schema_version, p.tier, len(p.rooms), 'room(s)')" out/<name>/plan.json
```

## When it goes wrong

Exit codes: 0 success (a run that finds no room still exits 0, with flags), 2 path does not exist, 3 `fetch-models`
failed, 1 any other error (a traceback).

| Symptom | Cause and fix |
|---|---|
| `ModelError: model 'depth-anything-v2-metric-indoor-small' is not usable ... Run: floorfathom fetch-models` | Model not fetched. Run `uv run floorfathom fetch-models`. Video keeps `sfm.pkl`, so a rerun resumes. |
| `ValueError: cannot tell the input tier of ...` | Empty folder or an unrecognised layout. Pass `--tier`. |
| `no photos found under ...` / `no video clips found in ...` | The folder has no stills or clips at the level the tier reads. |
| Room flagged `reference_ruler_not_found` / `reference_strip_not_found`, `scale_from_depth_model_only` | No ruler found. Not an error: the scale comes from the depth model with a 15% (video) or 30% (photo) scale interval. Recapture with the ruler flat, upright and in view. |
| `sfm_failed`, `sfm_registered_too_few_frames`, 0 rooms | SfM could not register the video. Recapture slowly with sideways steps and texture in view. |
| `image_unregistered:<file>` | Some photos could not be placed; the room is still built from the rest. Take more photos per room, in smaller steps (12-20). |
| `IndexError` in `run_photo_sfm`, no `plan.json` | Every photo in one room folder was unreadable. |
| `pycolmap` log lines on stderr | Normal in photo and video runs. |

Runs are deterministic for LiDAR, and for photo when the depth cache is present. Video results change between runs
(multithreaded COLMAP, depth maps, keyframe spacing) and so does the damage found on them.

## Models and third-party components

- Depth model: Depth Anything V2 Metric-Indoor Small, `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, revision
  `8078d68a9c75a972131914f6afd0c1723be0da7f`, files pinned by SHA-256 in `src/floorfathom/models.py`. Licence: Apache-2.0 per
  the upstream Depth-Anything-V2 repository; the `-hf` conversion declares none. Stored in `models/` (git-ignored), or
  `$FLOORFATHOM_MODELS`. Used by the photo and video tiers only; the LiDAR tier uses no learned model.
- Structure from motion: `pycolmap` (CPU). Damage classification is our own rules; a pretrained CLIP classifier was tried and dropped.
- Ruler scale, gravity, wall and opening detection, stitching, drift correction and uncertainty are our own code.

## How it works

The LiDAR tier is the core; the other tiers turn their input into a metric, gravity-aligned point cloud and reuse it.

1. `io_lidar.py`: read depth (mm), confidence, poses, intrinsics rescaled to the 256x192 depth grid.
2. `points.py`: back-project to a gravity-aligned world point cloud. Conventions were found by search, not assumed: depth
   as-is, OpenCV camera axes, quaternion is camera-to-world, world +y up. The picture in `rgb.mp4` looks rotated 90
   degrees; that is only the sensor's fixed landscape orientation.
3. `planes.py`: floor and ceiling are spikes in the height histogram; ceiling height is their distance. No ceiling spike
   means `null`, not an estimate.
4. `layout.py`, `estimate.py`: walls are cells occupied over most of 0.25-2.05 m of height (furniture is not). Free space is
   observed floor minus walls. Doorways are closed by extending wall segments across gaps up to 1.1 m, rooms are connected
   free space the camera visited, each outline is simplified and snapped to fitted wall lines. Right angles are never
   assumed. Doorways are gaps in the wall evidence a closure crosses.
5. `uncertainty.py`: intervals from a delete-2-of-20-chunks jackknife with a robust spread, plus an assumed systematic
   term. Rooms whose shape flips between replicates are flagged and their interval is stretched to cover the disagreement.
6. `report.py`, `schema.py`, `render.py`, `debug.py`, `cli.py`: contract, drawing, CLI.
7. `stitch.py`: each doorway is linked to the nearest other room outline within 1.5 m, overlapping rooms are found, the
   footprint is the union of the room polygons, and a global refinement (`refine_global`) adjusts placements. Rooms in their
   own frames (video, photo) are placed by gluing doorway to doorway (inward normals opposite, centres one assumed 0.15 m wall
   apart), most certain room first; a room with no doorway that fits is reported as unplaced, near-ties are flagged.
8. `drift.py`: per-chunk (x, z, yaw) drift from wall registration between time chunks of the walk, applied to every LiDAR run
   before the room estimate, with the on/off footprint ablation in `plan.json` (`stitching.drift`).
   `run_lidar(correct_drift=False)` switches it off (no CLI flag).

**Video tier** (`video_pipeline.py`): keyframes every 0.2 s (`io_video.py`), poses and sparse points by SfM (`sfm.py`), a dense
cloud from the depth model fitted to the SfM depths (`points_video.py`), gravity from the floor and ceiling (`world.py`).
SfM has no scale: the yellow ruler (`anchor.py`) is triangulated in the first 15 s and its tape length gives metres per SfM
unit. Then the LiDAR estimator with video-only inputs: camera-ray free space (`video_rays.py`), wall lines fitted to the whole
cloud (`video_walls.py`), floor and ceiling re-centred on their point plateau (`video_heights.py`) and through-ray doorway
detection (`video_openings.py`).

**Photo tier** (`photo_pipeline.py`): per-room registration of the stills (`photo_pose.py`), depth-model cloud
(`photo_scene.py`), ruler scale from several photos (`photo_reference.py`), wall planes by RANSAC with an evidence-weighted
Manhattan angle snap (`photo_layout.py`). Intervals come from leave-one-photo-out plus assumed systematic and scale terms.
Thin input gives wider intervals or null values, never a confident guess.

**Damage** (every tier, `assess.py`): per room each wall, the floor and the ceiling is unrolled into a flat colour patch
(`surfaces.py`), damage is found (`damage.py`: class-agnostic proposals from colour, thin ridges and depth relief, classified
by our own rules; classes are water stain, mould, structural crack, peeling paint, efflorescence, soot or fire, hole or impact,
sagging or bulging, other), then `room.damage`, `room.concealed_flags` (`concealed.py`) and `room.scope` (`scope.py`) are filled,
all with intervals.

## Known limitations

- Windows are not detected at the LiDAR tier (glass, partial-height gaps); only floor-level doorway gaps are. A closed door
  leaf hides its doorway.
- Wall length is corner to corner on the fitted lines; a wardrobe against a wall is measured to its face.
- Rooms are outlines of visited free space. Spaces that open onto the room through a gap wider than 1.1 m are merged with it.
- The systematic terms of the intervals (pose drift, plane offset) are assumptions in `uncertainty.py`, not yet calibrated
  against tape (only four rooms have tape truth).
- Drift is corrected in blocks of walk chunks, which can leave small steps at chunk boundaries. Whether the correction
  improves accuracy is untested without tape truth.
- Rooms placed by doorways assume a 0.15 m wall between them. Two doorways of equal width that fit either way are flagged
  `placement_ambiguous`. A room with no visible doorway cannot be placed (`stitching.unplaced`).
- Video walls still come out partly as fragments and the openings gate is not met; the video result changes between runs.
- Photo areas are null where some walls are never seen from the one spot; unseen walls get no length.
- Damage: hairline cracks (1-2 mm) are missed; door hardware, sockets, window bars, grain and glare still give false
  regions. On real LiDAR scans every reported region is a false positive: `CozmoData/` has no damage.

## Tests

`uv run pytest` runs 245 tests in about 6 minutes on synthetic captures with known geometry (`tests/synth.py`), ray-cast
into depth frames or rendered into stills and written in the real file layouts. They need no capture data, and no network
once `fetch-models` has run (a few tests load the real depth model). `floorfathom schema > schema/capture_plan.schema.json` regenerates the schema after changing
`schema.py` (a test checks they match).

## Layout

`src/floorfathom/` the package, `tests/` synthetic-capture tests, `scripts/` analyses and evaluation tools, `schema/` the
published JSON Schema, `capture_protocol.md` and `edge_cases.md` the capture docs, `ground_truth.json` tape measurements of
the flat captured in `Data/`, `docs/` design notes and `verification_notes.md`, `Deliverables/` the case-study documents.
Git-ignored and not in a clone: `Data/` (iPhone 16 captures), `CozmoData/` (Cozmo LiDAR samples), `models/` (depth model),
`out/` (outputs). `CLAUDE.md` holds the per-tier commands, timings and limits as the team keeps them.
