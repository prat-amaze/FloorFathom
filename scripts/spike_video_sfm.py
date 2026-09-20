"""Feasibility spike (throwaway): can SfM place the frames of an iPhone room video?

Picks a sharp frame every ~0.4 s, runs pycolmap SfM, prints how many keyframes got a
camera pose, how many 3D points came out, the reprojection error and the focal length
COLMAP settled on. Writes only under --work.

    uv run python scripts/spike_video_sfm.py Data/B1.MOV --work <scratch dir>
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def pick_keyframes(video: Path, out: Path, step_s: float, width: int) -> int:
    """Sharpest frame (variance of the Laplacian) in each window of ``step_s`` seconds."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    win = max(1, round(step_s * fps))
    out.mkdir(parents=True, exist_ok=True)
    best, best_score, n_saved, i = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            score = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if score > best_score:
                best, best_score = small, score
        i += 1
        if (i % win == 0 or not ok) and best is not None:
            cv2.imwrite(str(out / f"{n_saved:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_saved += 1
            best, best_score = None, -1.0
        if not ok:
            break
    cap.release()
    return n_saved


def pick_keyframes_flow(
    video: Path,
    out: Path,
    width: int,
    min_gap: int = 3,
    max_gap: int = 15,
    min_survive: float = 0.5,
    max_shift: float = 0.15,
    keep: int = 4,
) -> int:
    """Keyframes chosen by overlap: a new one when the tracked points of the last one are
    mostly lost (``min_survive``) or have moved far (``max_shift`` of the image width).
    The sharpest of the last ``keep`` frames is saved. ``max_gap`` frames without a trigger
    force one (blank walls have no points to track)."""
    cap = cv2.VideoCapture(str(video))
    out.mkdir(parents=True, exist_ok=True)
    flow_w = 320
    n_saved, since, ref_g, ref_pts, recent = 0, 0, None, None, []

    def corners(g):
        return cv2.goodFeaturesToTrack(g, maxCorners=300, qualityLevel=0.005, minDistance=6)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(gray, (flow_w, round(gray.shape[0] * flow_w / gray.shape[1])), interpolation=cv2.INTER_AREA)
        recent = (recent + [(cv2.Laplacian(gray, cv2.CV_64F).var(), small, g)])[-keep:]
        since += 1
        trigger = ref_g is None
        if not trigger and since >= min_gap:
            trigger = since >= max_gap
            if not trigger and ref_pts is not None and len(ref_pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_g, g, ref_pts, None)
                tracked = st.ravel() == 1
                shift = np.median(np.linalg.norm((nxt - ref_pts)[tracked], axis=-1)) / flow_w if tracked.any() else 1.0
                trigger = tracked.mean() < min_survive or shift > max_shift
        if trigger:
            _, best, best_g = max(recent, key=lambda r: r[0])
            cv2.imwrite(str(out / f"{n_saved:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_saved += 1
            ref_g, ref_pts, since, recent = best_g, corners(best_g), 0, []
    cap.release()
    return n_saved


def _ranges(ids: list[int]) -> str:
    """[0,1,2,5,6] -> '0-2, 5-6'"""
    out, start = [], 0
    for k in range(1, len(ids) + 1):
        if k == len(ids) or ids[k] != ids[k - 1] + 1:
            out.append(str(ids[start]) if start == k - 1 else f"{ids[start]}-{ids[k - 1]}")
            start = k
    return ", ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--step", type=float, default=0.4, help="seconds between keyframes")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--focal", type=float, default=0.6, help="initial focal length as a fraction of the long image side")
    ap.add_argument("--select", choices=["overlap", "time"], default="overlap", help="how keyframes are chosen")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="sequential")
    ap.add_argument("--window", type=int, default=12, help="sequential matcher: neighbours each keyframe is matched to")
    ap.add_argument("--peak", type=float, default=None, help="SIFT peak threshold (lower finds features on fainter texture)")
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--fix-focal", action="store_true", help="keep the initial focal length instead of refining it")
    args = ap.parse_args()

    work = args.work / args.video.stem
    if work.exists():
        shutil.rmtree(work)
    images, db, sparse = work / "images", work / "db.db", work / "sparse"
    sparse.mkdir(parents=True)

    t0 = time.perf_counter()
    if args.select == "overlap":
        n = pick_keyframes_flow(args.video, images, args.width)
    else:
        n = pick_keyframes(args.video, images, args.step, args.width)
    h = round(1920 * args.width / 1080)
    print(f"{args.video.name}: {n} keyframes at {args.width}x{h} ({time.perf_counter() - t0:.1f} s)")

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    f0 = args.focal * max(args.width, h)
    reader.camera_params = f"{f0},{args.width / 2},{h / 2},0"
    extraction = pycolmap.FeatureExtractionOptions()
    if args.peak is not None:
        extraction.sift.peak_threshold = args.peak
    if args.max_features is not None:
        extraction.sift.max_num_features = args.max_features
    pycolmap.extract_features(
        db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=extraction, device=pycolmap.Device.cpu,
    )
    t1 = time.perf_counter()
    print(f"features: {t1 - t0:.1f} s")
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = args.window
        pairing.loop_detection = False  # needs a vocabulary tree file; loop closure is a later step
        pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
    t2 = time.perf_counter()
    print(f"matching ({args.matcher}): {t2 - t1:.1f} s")
    opts = pycolmap.IncrementalPipelineOptions()
    if args.fix_focal:
        opts.ba_refine_focal_length = False
        opts.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, images, sparse, opts)
    print(f"mapping: {time.perf_counter() - t2:.1f} s, {len(recs)} model(s)")

    if not recs:
        print("RESULT: no model")
        return
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        cam = next(iter(rec.cameras.values()))
        print(
            f"model {k}: registered {rec.num_reg_images()}/{n} ({100 * rec.num_reg_images() / n:.0f}%), "
            f"{rec.num_points3D()} points, reproj {rec.compute_mean_reprojection_error():.2f} px, "
            f"mean track {rec.compute_mean_track_length():.1f}"
        )
        print(f"   camera {cam.model.name} params (f, cx, cy, k) = {np.round(cam.params, 3).tolist()}  (initial f {f0:.0f})")
        print(f"   keyframes (time order): {_ranges(sorted(int(rec.images[i].name[:5]) for i in rec.reg_image_ids()))}")


if __name__ == "__main__":
    main()
