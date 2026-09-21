"""Floor and ceiling of a depth-model room: reflections under the floor, a bowed ceiling with a long upper tail."""

from __future__ import annotations

import numpy as np

from floorfathom.planes import find_floor
from floorfathom.video_heights import floor_from_strongest_spike, refine_heights

FLOOR, CEILING = 0.0, 2.79


def _floor_points(rng, n=4000) -> np.ndarray:
    return rng.normal(FLOOR, 0.015, n)


def _reflection_tail(rng, n=2500) -> np.ndarray:
    """A glossy floor returns the depth of its mirror image: points spread below the true floor."""
    return rng.uniform(-0.30, -0.03, n)


def _bowed_ceiling(rng, n=8000) -> np.ndarray:
    """Seen at a slant the ceiling bows away from the camera: a plateau at the true height plus a long tail above it."""
    return CEILING + np.concatenate([rng.normal(0.0, 0.03, n // 2), rng.exponential(0.09, n - n // 2)])


def _walls(rng, n=20000) -> np.ndarray:
    return rng.uniform(0.3, 2.4, n)


def test_a_reflection_tail_does_not_pull_the_floor_down():
    rng = np.random.default_rng(0)
    y = np.concatenate([_floor_points(rng), _reflection_tail(rng), _walls(rng), _bowed_ceiling(rng)])
    r = refine_heights(y, floor_y=-0.12, ceiling_y=CEILING + 0.12)  # first guesses pulled toward the tail and the top of the plateau
    assert r.refined
    assert abs(r.floor_y - FLOOR) < 0.03


def test_the_ceiling_moves_from_the_top_of_a_skewed_plateau_toward_its_centre():
    rng = np.random.default_rng(1)
    y = np.concatenate([_floor_points(rng), _walls(rng), _bowed_ceiling(rng)])
    guess = CEILING + 0.20  # the highest spike of a smeared ceiling is a noise peak on its upper side
    r = refine_heights(y, floor_y=FLOOR, ceiling_y=guess)
    assert abs(r.ceiling_y - CEILING) < abs(guess - CEILING) / 2
    assert abs(r.ceiling_height - CEILING) < 0.08
    assert 0.02 < r.ceiling_spread < 0.12  # the real uncertainty of a depth-model ceiling, for the caller to widen its interval


def test_a_spike_beyond_the_window_is_not_mistaken_for_the_ceiling():
    rng = np.random.default_rng(2)
    y = np.concatenate([_floor_points(rng), _bowed_ceiling(rng), np.full(30000, CEILING + 0.9)])  # e.g. the ceiling of the next room
    r = refine_heights(y, floor_y=FLOOR, ceiling_y=CEILING + 0.05)
    assert abs(r.ceiling_y - CEILING) < 0.08


def test_too_few_points_keep_the_first_guess_and_say_so():
    y = np.array([0.0, 0.01, 2.8, 2.79] * 20)
    r = refine_heights(y, floor_y=0.05, ceiling_y=2.85)
    assert not r.refined
    assert (r.floor_y, r.ceiling_y) == (0.05, 2.85)


def test_a_full_glossy_reflection_below_the_floor_fools_find_floor_but_not_the_spike_scan():
    """A glossy tile floor mirrors the whole room below itself: a strong reflected-ceiling spike 2.79 m under the
    true floor, and reflected walls filling the gap. find_floor locks onto that spike; the spike scan does not."""
    rng = np.random.default_rng(4)
    floor = _floor_points(rng, 6000)
    ceiling = rng.normal(CEILING, 0.02, 4000)
    walls = _walls(rng, 8000)
    refl_ceiling = rng.normal(-CEILING, 0.02, 4000)  # the mirror image of the ceiling, below the floor
    refl_walls = rng.uniform(-CEILING, -0.05, 8000)
    y = np.concatenate([floor, ceiling, walls, refl_ceiling, refl_walls])
    assert find_floor(y).y < -1.0  # the reflection tail drags the naive search window below the true floor
    assert abs(floor_from_strongest_spike(y) - FLOOR) < 0.05  # the strongest spike is still the true floor


def test_the_spike_scan_finds_the_floor_of_an_ordinary_room():
    rng = np.random.default_rng(5)
    y = np.concatenate([_floor_points(rng), _walls(rng), _bowed_ceiling(rng)])
    assert abs(floor_from_strongest_spike(y) - FLOOR) < 0.05


def test_no_ceiling_guess_gives_no_ceiling():
    rng = np.random.default_rng(3)
    r = refine_heights(np.concatenate([_floor_points(rng), _walls(rng)]), floor_y=0.02, ceiling_y=None)
    assert r.ceiling_y is None and r.ceiling_height is None
    assert abs(r.floor_y - FLOOR) < 0.03
