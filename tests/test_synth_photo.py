import cv2
import numpy as np

from synth import checkerboard_texture, textured_room


def test_checkerboard_texture_has_sift_keypoints():
    tex = checkerboard_texture("brick", cell_m=0.25, mpp=0.01)
    ss, hh = np.meshgrid(np.linspace(0, 4, 400), np.linspace(0, 2.6, 260))
    colours = tex(ss.ravel(), hh.ravel())
    img = (np.clip(colours, 0, 1).reshape(260, 400, 3) * 255).astype(np.uint8)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp = cv2.SIFT_create().detect(grey, None)
    assert len(kp) > 200


def test_textured_room_path_actually_translates():
    stills = textured_room(
        walls="rect4x3", height=2.6,
        path=[(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)],
        photo_positions=9,
    )
    positions = np.array([s.position for s in stills])
    spread = positions[:, [0, 2]].max(axis=0) - positions[:, [0, 2]].min(axis=0)
    assert spread.min() > 1.0
