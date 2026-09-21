"""Tests for run_photo_sfm: translating stills register, EXIF focal missing is flagged."""
import numpy as np
from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pose import run_photo_sfm
from synth import FLOOR_Y, _pose, _raycast_textured, _sinusoidal_sh_tex, rect


def _photoset() -> PhotoSet:
    """8 photos from a lateral sweep along x, all facing the north wall at z=3.

    This gives:
      * large view overlap between consecutive cameras (same yaw=0)
      * non-planar 3D structure (north wall + left/right side walls)
      * real camera translation (x-spread = 3 m)
    — the conditions pycolmap's SfM needs to register all frames.
    """
    walls = rect(0.0, 0.0, 4.0, 3.0)
    tex = _sinusoidal_sh_tex(seed=42)
    xs = np.linspace(0.5, 3.5, 8)
    images = []
    for k, x in enumerate(xs):
        o = np.array([x, 0.0, 0.5])
        R = _pose(o, 0.0, 0.0)  # yaw=0: face north wall
        rgb = _raycast_textured(o, R, walls, 2.6, FLOOR_Y, tex)
        images.append(PhotoImage(f"synth_{k:03d}", sha256=f"s{k}", rgb=rgb, full_size=(320, 240), f35=24.0, f_px=266.0))
    return PhotoSet(room="synth", images=images)


def test_translating_stills_register_with_real_baseline(tmp_path):
    photos = _photoset()
    sfm = run_photo_sfm(photos, tmp_path / "work", seed=0)
    assert sfm is not None
    assert sfm.registered_fraction >= 0.75
    centres = sfm.centers[sfm.registered]
    spread = centres[:, [0, 2]].max(axis=0) - centres[:, [0, 2]].min(axis=0)
    assert spread.min() > 0.05


def test_missing_exif_focal_falls_back_and_flags(tmp_path):
    photos = _photoset()
    for im in photos.images:
        im.f35 = None
    sfm = run_photo_sfm(photos, tmp_path / "work2", seed=0)
    assert sfm is not None
    assert "focal_prior_default" in sfm.flags
