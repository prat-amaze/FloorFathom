"""Camera rays and wall lines for the video tier: a 5 x 4 m room whose floor gave no depth points (glossy floor)."""

from __future__ import annotations

import numpy as np

from floorfathom import layout as L
from floorfathom.estimate import Params, estimate
from floorfathom.video_rays import RayBundle, carve, seen_from_votes

W, D, H = 5.0, 4.0, 2.6  # room x in [0, W], z in [0, D], floor y = 0, ceiling y = H
CAM_Y = 1.4
ROOM_AREA = W * D


def _hit(o: np.ndarray, angle: float) -> float:
    """Distance along azimuth ``angle`` from o = (x, z) to the room's walls."""
    d = np.array([np.cos(angle), np.sin(angle)])
    ts = []
    for axis, lo, hi in ((0, 0.0, W), (1, 0.0, D)):
        if abs(d[axis]) > 1e-9:
            ts += [t for t in ((lo - o[axis]) / d[axis], (hi - o[axis]) / d[axis]) if t > 0]
    return min(ts)


def _rays(origins: list[tuple[float, float]], per_frame: int = 240, seed: int = 0) -> RayBundle:
    rng = np.random.default_rng(seed)
    ends = []
    for x, z in origins:
        a = rng.uniform(0, 2 * np.pi, per_frame)
        e = []
        for ang in a:
            r = _hit(np.array([x, z]), ang)
            elev = rng.uniform(-0.6, 0.6)  # up and down: the surface a pixel shows is on the wall, floor or ceiling
            e.append([x + r * np.cos(ang), CAM_Y + r * np.tan(elev), z + r * np.sin(ang)])
        e = np.array(e)
        e[:, 1] = np.clip(e[:, 1], 0.0, H)
        ends.append(e.astype(np.float32))
    org = np.array([[x, CAM_Y, z] for x, z in origins])
    chunk = (np.arange(len(origins)) * 4 // len(origins)).astype(np.int16)
    return RayBundle(org, ends, chunk, 4)


def _walk() -> list[tuple[float, float]]:
    return [(2.0 + 0.15 * i, 1.6 + 0.1 * (i % 5)) for i in range(20)]


def _wall_points() -> np.ndarray:
    """Dense points on the four walls between 0.3 m and the ceiling; nothing on the floor."""
    rng = np.random.default_rng(1)
    pts = []
    for _ in range(8000):
        y = rng.uniform(0.3, H)
        t = rng.uniform(0, 2 * (W + D))
        if t < W:
            pts.append([t, y, 0.0])
        elif t < W + D:
            pts.append([W, y, t - W])
        elif t < 2 * W + D:
            pts.append([2 * W + D - t, y, D])
        else:
            pts.append([0.0, y, 2 * (W + D) - t])
    return np.array(pts, np.float32)


def _grid() -> L.Grid:
    return L.Grid.around(np.array([[0.0, 0.0], [W, D]]), margin=0.5)


def test_rays_mark_the_air_of_the_room_and_stop_at_its_walls():
    grid = _grid()
    votes = carve(_rays(_walk()), np.eye(3), 1.0, grid, floor_y=0.0)
    seen = seen_from_votes(votes)
    inside = np.zeros_like(seen)
    x0, z0 = np.floor((np.array([0.4, 0.4]) - [grid.x0, grid.z0]) / grid.cell).astype(int)
    x1, z1 = np.floor((np.array([W - 0.4, D - 0.4]) - [grid.x0, grid.z0]) / grid.cell).astype(int)
    inside[z0:z1, x0:x1] = True
    assert seen[inside].mean() > 0.9  # the interior is seen although no point lies on the floor
    ix, iz = np.meshgrid(np.arange(grid.nx), np.arange(grid.nz))
    xz = grid.to_xz(ix, iz)
    outside = (xz[..., 0] < -0.1) | (xz[..., 0] > W + 0.1) | (xz[..., 1] < -0.1) | (xz[..., 1] > D + 0.1)
    assert not (seen & outside).any()  # rays never leave the room: they stop short of the walls


def test_a_ray_ending_below_the_floor_is_cut_at_the_floor():
    # a reflection on a glossy floor: the depth model puts the surface 3 m below the floor, far past the wall
    grid = _grid()
    o = np.array([[2.5, CAM_Y, 2.0]])
    end = np.array([[2.5 + 8.0, -3.0, 2.0]], np.float32)  # would reach x = 10.5 m, outside the room
    votes = carve(RayBundle(o, [end], np.zeros(1, np.int16), 1), np.eye(3), 1.0, grid, floor_y=0.0)
    ix, iz = np.meshgrid(np.arange(grid.nx), np.arange(grid.nz))
    xz = grid.to_xz(ix, iz)
    reach = float(xz[(votes[0] > 0)][:, 0].max())
    # the ray meets the floor plane after 1.4 / 4.4 of its length: 2.5 + 8 * 1.4 / 4.4 m, minus the margin
    assert 4.6 < reach < 5.0 + 0.1


def test_the_estimator_finds_the_room_with_rays_and_fragments_without_them():
    pts, grid = _wall_points(), _grid()
    traj = np.array(_walk())[:, ::1]
    without = estimate(pts, traj, Params(), grid=grid)
    hint = seen_from_votes(carve(_rays(_walk()), np.eye(3), 1.0, grid, floor_y=0.0))
    with_rays = estimate(pts, traj, Params(), grid=grid, free_hint=hint)
    assert with_rays.rooms
    best = max(with_rays.rooms, key=lambda r: r.area)
    assert abs(best.area - ROOM_AREA) < 0.06 * ROOM_AREA
    assert not without.rooms or max(r.area for r in without.rooms) < 0.7 * ROOM_AREA  # what the rays fix


def test_free_space_is_cut_behind_a_wall_line_but_not_at_a_line_the_camera_stands_near():
    grid = _grid()
    free = np.zeros((grid.nz, grid.nx), bool)
    ix, iz = np.meshgrid(np.arange(grid.nx), np.arange(grid.nz))
    xz = grid.to_xz(ix, iz)
    free[(xz[..., 0] > -0.4) & (xz[..., 0] < W + 2.5) & (xz[..., 1] > 0.2) & (xz[..., 1] < D - 0.2)] = True  # room + a balcony seen through the wall x = W
    traj = np.array(_walk())
    wall = L.WallLine(np.array([1.0, 0.0]), W, 0.0, D)  # the line x = W, seen from z = 0 to D
    cut = L.clip_to_walls(free, grid, [wall], traj)
    assert cut[(xz[..., 0] < W - 0.1) & free].all()  # the room stays
    assert not cut[(xz[..., 0] > W + 0.1)].any()  # the balcony behind the wall goes
    through = L.WallLine(np.array([1.0, 0.0]), 2.2, 0.0, D)  # a "wall" through the camera path is furniture or a misfit
    assert (L.clip_to_walls(free, grid, [through], traj) == free).all()
