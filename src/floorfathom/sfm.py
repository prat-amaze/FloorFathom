"""Structure-from-Motion: keyframes to camera poses and a sparse 3D point cloud.

Feature points seen in several frames are triangulated, and every camera pose and every 3D
point is solved together (pycolmap). The scale of the result is arbitrary: the room comes
out with the right shape but in unknown units (see ``world.py`` for metres).

Choices taken from the feasibility spike on the real clips (``scripts/spike_video_sfm.py``):
  * the focal length starts from the clip's own metadata and is then refined. The phone
    writes a 35 mm-equivalent focal length into each clip, but as a rounded whole number
    and without the video stabilisation crop, so it only gives a starting value: clips
    from one phone read 16, 16, 16, 14, 15 and 14 mm and settled at about 798 px (16) and
    728 px (14) at 720 px wide. Three separate H1 models agreed within 0.6% from the same
    start, and forcing the wrong value (798 px on a 14 mm clip) raised the reprojection
    error by about 15%. A refined value far from the start is flagged. Distortion is fixed
    at zero: refined values came out at 0.00.
  * sequential matching gives the same models as all-pairs at a fraction of the time,
  * a lower SIFT peak threshold finds more features on faint texture.
A capture whose reconstruction splits (fast pan, blur, or a blank wall filling the view)
is reported, not repaired: only the largest connected piece is kept and the time spans
without a pose are listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from .io_video import Keyframes, read_camera_metadata

# Starting focal length: (35 mm-equivalent in mm) x FOCAL_PER_MM x (long image side). Measured on
# iPhone 16 video: 16 mm settled at 0.0390 and 14 mm at 0.0406 of the long side per mm (the
# phone rounds the metadata to whole mm), so the mean is used. It is a start, not a result.
FOCAL_PER_MM = 0.0398
DEFAULT_FOCAL_FRACTION = 0.60  # of the long image side, when the clip has no metadata
FOCAL_RATIO_OK = (0.85, 1.18)  # refined / starting focal outside this is flagged
MIN_REGISTERED = 0.6  # below this the reconstruction is not used at all


@dataclass
class SfmResult:
    registered: np.ndarray  # (N,) bool, keyframe got a pose in the kept model
    centers: np.ndarray  # (N, 3) camera centres, SfM units (nan where not registered)
    cam_to_world: np.ndarray  # (N, 3, 3) rotation, camera axes x right, y down, z forward (nan where not registered)
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy in keyframe pixels
    image_size: tuple[int, int]
    points: np.ndarray  # (M, 3) sparse points, SfM units
    track_length: np.ndarray  # (M,) number of keyframes that see each point
    point_error: np.ndarray  # (M,) reprojection error in pixels
    n_models: int
    mean_reprojection_px: float
    unposed_spans: list[tuple[float, float]] = field(default_factory=list)  # seconds without a pose
    flags: list[str] = field(default_factory=list)
    focal_prior_px: float = 0.0  # the starting focal length, keyframe pixels
    focal_source: str = "unknown"  # "metadata_35mm_equivalent" or "default"
    # per registered keyframe: pixels (u, v) of its triangulated points and their depth along the
    # camera z axis in SfM units, used to fit a depth model's output to the reconstruction
    observations: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def registered_fraction(self) -> float:
        return float(self.registered.mean())


def _spans(mask: np.ndarray, time_s: np.ndarray) -> list[tuple[float, float]]:
    """Time ranges of consecutive True entries."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            out.append((round(float(time_s[start]), 2), round(float(time_s[end]), 2)))
            start = None
    return out


def focal_prior(kf: Keyframes) -> tuple[float, str]:
    """Starting focal length in keyframe pixels, and where it came from."""
    long_side = max(kf.size)
    f35 = read_camera_metadata(kf.video).focal_35mm_equivalent
    if f35 is None:
        return DEFAULT_FOCAL_FRACTION * long_side, "default"
    return f35 * FOCAL_PER_MM * long_side, "metadata_35mm_equivalent"


def run_sfm(kf: Keyframes, work: str | Path, seed: int = 0, window: int = 12) -> SfmResult | None:
    """Reconstruct the keyframes. Returns None when no model could be built at all."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    db, sparse = work / "db.db", work / "sparse"
    if db.exists():
        db.unlink()
    sparse.mkdir(exist_ok=True)
    w, h = kf.size
    f, focal_source = focal_prior(kf)

    pycolmap.set_random_seed(seed)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{f},{w / 2},{h / 2},0"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = 0.004
    extraction.sift.max_num_features = 8192
    cpu = pycolmap.Device.cpu
    pycolmap.extract_features(
        db, kf.directory, image_names=kf.names, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction, device=cpu,
    )
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = window
    pairing.loop_detection = False  # needs a vocabulary tree file; loop closure is a later step
    pycolmap.match_sequential(db, pairing_options=pairing, device=cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.ba_refine_extra_params = False  # distortion stays at zero
    recs = pycolmap.incremental_mapping(db, kf.directory, sparse, opts)
    if not recs:
        return None

    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    n = len(kf)
    index = {name: i for i, name in enumerate(kf.names)}
    registered = np.zeros(n, dtype=bool)
    centers = np.full((n, 3), np.nan)
    rot = np.full((n, 3, 3), np.nan)
    observations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for image_id in rec.reg_image_ids():
        img = rec.images[image_id]
        i = index[img.name]
        cam_from_world = img.cam_from_world()
        world_from_cam = cam_from_world.inverse()
        registered[i] = True
        centers[i] = world_from_cam.translation
        rot[i] = world_from_cam.rotation.matrix()
        r_cw, t_cw = cam_from_world.rotation.matrix(), np.asarray(cam_from_world.translation)
        uv, z = [], []
        for p in img.points2D:
            if p.has_point3D():
                depth = float((r_cw @ rec.points3D[p.point3D_id].xyz + t_cw)[2])
                if depth > 0:
                    uv.append(p.xy)
                    z.append(depth)
        observations[i] = (np.array(uv).reshape(-1, 2), np.array(z))
    cam = next(iter(rec.cameras.values()))
    params = cam.params  # SIMPLE_RADIAL: f, cx, cy, k
    pts = list(rec.points3D.values())

    result = SfmResult(
        registered=registered,
        centers=centers,
        cam_to_world=rot,
        intrinsics=(float(params[0]), float(params[0]), float(params[1]), float(params[2])),
        image_size=(w, h),
        points=np.array([p.xyz for p in pts]).reshape(-1, 3),
        track_length=np.array([p.track.length() for p in pts], dtype=int),
        point_error=np.array([p.error for p in pts], dtype=float),
        n_models=len(recs),
        mean_reprojection_px=float(rec.compute_mean_reprojection_error()),
        unposed_spans=_spans(~registered, kf.time_s),
        focal_prior_px=f,
        focal_source=focal_source,
        observations=observations,
    )
    if focal_source == "default":
        result.flags.append("focal_prior_default")
    if not FOCAL_RATIO_OK[0] <= result.intrinsics[0] / f <= FOCAL_RATIO_OK[1]:
        result.flags.append("focal_far_from_metadata")
    if result.registered_fraction < MIN_REGISTERED:
        result.flags.append("sfm_registered_too_few_frames")
    elif result.unposed_spans:
        result.flags.append("sfm_capture_break")
    if len(recs) > 1:
        result.flags.append("sfm_split_into_several_models")
    return result
