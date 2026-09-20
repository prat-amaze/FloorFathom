"""Scale anchor: triangulating a reference segment and finding a white strip in an image."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from floorfathom.anchor import SegmentObs, estimate_anchor, find_strip, track_strip
from floorfathom.io_video import Keyframes
from floorfathom.sfm import SfmResult

F, CX, CY = 700.0, 360.0, 640.0
TRUE_LENGTH_M = 0.20
SCALE = 0.37  # metres per SfM unit, arbitrary
TOP, BOTTOM = np.array([0.10, -0.10, 1.5]), np.array([0.10, -0.10 + TRUE_LENGTH_M / SCALE / 4, 1.5])


def _sfm(centres: np.ndarray) -> SfmResult:
    n = len(centres)
    return SfmResult(
        registered=np.ones(n, dtype=bool),
        centers=centres,
        cam_to_world=np.repeat(np.eye(3)[None], n, axis=0),  # every camera faces +z
        intrinsics=(F, F, CX, CY),
        image_size=(720, 1280),
        points=np.zeros((1, 3)),
        track_length=np.ones(1, dtype=int),
        point_error=np.zeros(1),
        n_models=1,
        mean_reprojection_px=0.5,
    )


def _observe(sfm: SfmResult, noise_px: float, seed: int, top=TOP, bottom=BOTTOM) -> list[SegmentObs]:
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(sfm.centers):
        px = []
        for x in (top, bottom):
            p = x - c
            px.append((F * p[0] / p[2] + CX + rng.normal(0, noise_px), F * p[1] / p[2] + CY + rng.normal(0, noise_px)))
        out.append(SegmentObs(i, px[0], px[1]))
    return out


def _true_length_sfm(top=TOP, bottom=BOTTOM) -> float:
    return float(np.linalg.norm(top - bottom))


def test_scale_recovered_when_the_camera_moves_sideways():
    sfm = _sfm(np.column_stack([np.linspace(-0.6, 0.6, 60), np.zeros(60), np.zeros(60)]))
    anchor = estimate_anchor(sfm, _observe(sfm, 0.5, seed=1), true_length_m=TRUE_LENGTH_M)
    assert anchor is not None and anchor.flags == []
    assert anchor.length_sfm == pytest.approx(_true_length_sfm(), rel=0.02)
    assert anchor.scale == pytest.approx(TRUE_LENGTH_M / _true_length_sfm(), rel=0.02)
    assert anchor.max_ray_angle_deg > 20
    assert anchor.rel_sigma < 0.05


def test_camera_that_hardly_moves_is_flagged_and_uncertain():
    sfm = _sfm(np.column_stack([np.linspace(-0.004, 0.004, 60), np.zeros(60), np.zeros(60)]))  # 8 mm of travel
    anchor = estimate_anchor(sfm, _observe(sfm, 0.5, seed=2), true_length_m=TRUE_LENGTH_M)
    assert anchor is not None
    assert "anchor_weak_baseline" in anchor.flags
    assert anchor.rel_sigma > 0.2  # the interval must say the scale is not pinned down


def test_too_few_observations_gives_no_anchor():
    sfm = _sfm(np.column_stack([np.linspace(-0.6, 0.6, 2), np.zeros(2), np.zeros(2)]))
    assert estimate_anchor(sfm, _observe(sfm, 0.5, seed=3)) is None


def _door_image(strip: bool = True) -> np.ndarray:
    img = np.full((1280, 720, 3), (40, 85, 140), np.uint8)  # brown door, BGR
    img += np.random.default_rng(0).integers(0, 12, img.shape, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (150, 500), (240, 240, 240), -1)  # a large white wall patch
    if strip:
        cv2.rectangle(img, (340, 400), (372, 700), (235, 235, 235), -1)  # 32 x 300 px white strip
        for y in range(420, 700, 60):
            cv2.line(img, (345, y), (367, y), (60, 60, 60), 2)  # dark tick marks
    return img


def test_finds_the_white_strip_but_not_the_white_wall():
    top, bottom = find_strip(_door_image())
    assert top[1] < bottom[1]
    assert top[0] == pytest.approx(356, abs=3) and bottom[0] == pytest.approx(356, abs=3)
    assert top[1] == pytest.approx(400, abs=4) and bottom[1] == pytest.approx(700, abs=4)


def test_follows_the_strip_from_where_it_was_last_seen():
    first = find_strip(_door_image())
    mid = tuple(0.5 * (np.asarray(first[0]) + np.asarray(first[1])))
    shifted = np.roll(_door_image(), 40, axis=1)  # the strip has moved 40 px to the right
    top, bottom = find_strip(shifted, near=mid)
    assert top[0] == pytest.approx(396, abs=3) and bottom[0] == pytest.approx(396, abs=3)
    assert find_strip(shifted, near=(50.0, 1200.0)) is None  # nothing white and strip-like near there


def test_no_strip_no_detection():
    assert find_strip(_door_image(strip=False)) is None


def _ruler_image() -> np.ndarray:
    img = _door_image(strip=False)  # brown door with a large white wall patch
    cv2.rectangle(img, (340, 400), (372, 700), (40, 235, 215), -1)  # BGR yellow-green ruler body (hue about 33 as measured), 32 x 300 px
    cv2.rectangle(img, (330, 400), (340, 700), (200, 200, 200), -1)  # its clear plastic edge, light grey
    for y in range(420, 700, 60):
        cv2.line(img, (345, y), (367, y), (30, 30, 30), 2)  # black tick marks
    return img


def test_finds_the_yellow_ruler_and_not_the_white_wall_or_its_clear_edge():
    top, bottom = find_strip(_ruler_image(), colour="yellow")
    assert top[0] == pytest.approx(356, abs=3) and bottom[0] == pytest.approx(356, abs=3)
    assert top[1] == pytest.approx(400, abs=4) and bottom[1] == pytest.approx(700, abs=4)


def test_a_yellow_search_ignores_a_white_strip():
    assert find_strip(_door_image(), colour="yellow") is None


def test_video_yellow_is_more_orange_than_still_yellow():
    img = _door_image(strip=False)
    cv2.rectangle(img, (340, 400), (372, 700), (30, 175, 200), -1)  # BGR, hue about 25: how video frames render the ruler
    assert find_strip(img, colour="yellow") is None
    top, bottom = find_strip(img, colour="yellow_video")
    assert top[1] == pytest.approx(400, abs=4) and bottom[1] == pytest.approx(700, abs=4)


def test_an_unknown_colour_is_an_error():
    with pytest.raises(ValueError):
        find_strip(_ruler_image(), colour="green")


def test_the_strip_search_stops_after_the_time_limit(tmp_path):
    n = 6
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"{i}.png"), _door_image())
    kf = Keyframes(tmp_path, [f"{i}.png" for i in range(n)], np.arange(n), np.arange(n, dtype=float), np.ones(n), (720, 1280), 30.0, tmp_path / "c.MOV")
    sfm = _sfm(np.zeros((n, 3)))
    assert len(track_strip(kf, sfm)) == n
    assert [o.frame for o in track_strip(kf, sfm, until_s=2.5)] == [0, 1, 2]  # keyframes at 0, 1 and 2 s
