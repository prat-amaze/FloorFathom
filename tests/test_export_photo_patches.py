"""The patch export used to build the damage detector's real-photo test set, run on the synthetic one-spot room."""

import sys
from pathlib import Path

import numpy as np
from test_photo_damage import MPP, _photo_room, _quads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_photo_patches import export_patches  # noqa: E402


def test_every_wall_and_the_ceiling_are_saved_with_their_pixels_and_scale(tmp_path):
    stems = export_patches(_photo_room(_quads()), tmp_path, mpp=MPP)
    assert sorted(stems) == ["room_0_r0_w0", "room_0_r0_w1", "room_0_r0_w2", "room_0_r0_w3", "room_0_room_0_ceiling"]
    means = {}
    for s in stems:
        z = np.load(tmp_path / f"{s}.npz")
        assert z["rgb"].dtype == np.float16 and 0 <= z["rgb"].min() and z["rgb"].max() <= 1 and z["valid"].dtype == bool and float(z["m_per_px"]) == MPP
        assert z["valid"].mean() > 0.3 and (tmp_path / f"{s}.png").exists()
        means[str(z["surface"])] = z
    stained = means["r0_w0"]  # the stain sits at s 1.8 m, h 1.2 m, i.e. column 180, row 120 at 1 cm per pixel
    assert stained["valid"][120, 180] and stained["rgb"][120, 180, 2] < 0.8 * np.median(means["r0_w1"]["rgb"][..., 2].astype(np.float32))  # yellow-brown: little blue
