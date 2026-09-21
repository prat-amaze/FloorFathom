"""Floor and ceiling heights of a video room, from where most of their points lie rather than from the highest spike.

A LiDAR ceiling is a thin sheet: its points fall in one or two 1 cm bins, and the highest strong spike of the
height histogram is the ceiling (``planes.find_ceiling``). A depth model's ceiling is not: it is seen at a slant
from a walking camera, so its depth is off by a few percent that grow with range and the sheet comes out bowed.
On H1 the ceiling points spread over 2.4-3.2 m above the floor, with a broad plateau at 2.74-2.90 m. The
highest spike of such a plateau is whichever noise peak sits on its upper side, and it read 2.91 m against a tape
of 2.79 m. The same plateau's centre is stable: a bow that is not skewed puts as many points above the true
height as below it, so the centre of mass is unbiased where the top edge is biased high.

``refine_heights`` therefore takes the first-guess planes of the estimator and re-centres each on the densest
10 cm of the points around it (the mode of the histogram smoothed over 10 cm, then the median of the points
within it). It also returns the spread of the points that support each plane: for a depth-model ceiling this is
the real uncertainty (several cm), and the caller should widen the interval with it instead of quoting the
1 cm bin of the LiDAR tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN = 0.01
WINDOW = 0.35  # only points within this of the first guess are used: a wrong guess is nudged, never replaced by another surface
KERNEL = 0.10  # the histogram is smoothed over this: wider than a noise peak, narrower than the bow
MIN_POINTS = 300  # fewer points than this in the window and the first guess is kept
SPREAD_WINDOW = 0.15  # the spread is measured on the plateau around the centre, not on the walls and furniture the window also holds


@dataclass
class Heights:
    floor_y: float
    ceiling_y: float | None
    floor_spread: float  # robust std (m) of the points that support the floor, 0 when the guess was kept
    ceiling_spread: float
    refined: bool  # False when there were too few points and the first guesses were returned unchanged

    @property
    def ceiling_height(self) -> float | None:
        return None if self.ceiling_y is None else self.ceiling_y - self.floor_y


def _centre(y: np.ndarray, guess: float, window: float, kernel: float, min_points: int) -> tuple[float, float, bool]:
    """(centre, spread, refined) of the densest ``kernel`` of the points within ``window`` of ``guess``."""
    sel = y[np.abs(y - guess) <= window]
    if len(sel) < min_points:
        return guess, 0.0, False
    edges = np.arange(guess - window, guess + window + BIN, BIN)
    hist, edges = np.histogram(sel, bins=edges)
    smooth = np.convolve(hist, np.ones(int(round(kernel / BIN))), mode="same")
    peak = 0.5 * (edges[int(smooth.argmax())] + edges[int(smooth.argmax()) + 1])
    core = sel[np.abs(sel - peak) <= kernel / 2 + BIN]
    if len(core) < min_points // 3:
        return guess, 0.0, False
    centre = float(np.median(core))
    near = sel[np.abs(sel - centre) <= SPREAD_WINDOW]
    spread = 1.4826 * float(np.median(np.abs(near - np.median(near))))
    return centre, spread, True


def refine_heights(
    y: np.ndarray,
    floor_y: float,
    ceiling_y: float | None,
    window: float = WINDOW,
    kernel: float = KERNEL,
    min_points: int = MIN_POINTS,
) -> Heights:
    """Floor and ceiling re-centred on where their points lie.

    ``y`` are the heights (m, world +y up) of the room's points, ``floor_y`` and ``ceiling_y`` the estimator's first
    guesses (``ceiling_y`` may be None: then none is returned). Each plane may move by at most ``window``; the
    spreads are the robust std of the points within ``SPREAD_WINDOW`` of each plane and are meant to widen the intervals.
    """
    y = np.asarray(y, dtype=float)
    f, fs, f_ok = _centre(y, floor_y, window, kernel, min_points)
    if ceiling_y is None:
        return Heights(f, None, fs, 0.0, f_ok)
    c, cs, c_ok = _centre(y, ceiling_y, window, kernel, min_points)
    return Heights(f, c, fs, cs, f_ok and c_ok)
