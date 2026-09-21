import numpy as np

from floorfathom.drift import _apply
from floorfathom.lidar_frames import make_undo


def test_undo_inverts_the_drift_correction_of_a_chunk():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, (200, 3)).astype(np.float32)
    shift = np.array([0.12, -0.07, np.radians(2.5)])
    centre = np.array([1.3, -0.8])
    xz = _apply(pts[:, [0, 2]], np.tile(shift, (len(pts), 1)), centre)
    corrected = pts.copy()
    corrected[:, 0], corrected[:, 2] = xz[:, 0], xz[:, 1]
    back = make_undo(shift, centre)(corrected)
    assert np.abs(back - pts).max() < 1e-4
    assert np.array_equal(back[:, 1], pts[:, 1])  # height is untouched
    assert np.abs(corrected - pts).max() > 0.05  # the correction did move things
