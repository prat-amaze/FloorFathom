"""Damage on the video tier: the frame adapter in front of ``assess.assess_room``.

Unrolling surfaces, detection, classes, concealed-damage flags and scope items are shared with the other tiers
(``surfaces.py``, ``damage.py``, ``concealed.py``, ``scope.py``, ``assess.py``). What is specific to video lives here:

* poses come from SfM in SfM units and are moved into the plan's frame with the same gravity rotation and metric
  scale as the point cloud, so a region's size is in metres by the same route as a wall length; the scale's
  relative error goes into every interval (``scale_rel_sigma``);
* only a handful of keyframes are used per surface, picked greedily so that together they see most of it face on;
* the depth model is used to tell what stands in front of a surface (furniture, a person), never for relief: its
  depth is only good to several percent, more than the 2.5 cm relief the detector looks for;
* the pictures are the 720 px keyframes. Full-resolution frames from the clip were tried and dropped: on synthetic
  clips they found no thinner crack than the keyframes (a 3 mm crack is found, 1 and 2 mm are not, at either size) and
  added 8 to 54 spurious regions, because a 5 mm patch pixel sampled from finer frames is not averaged and so
  carries their sensor noise (``scripts/eval_damage_video.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from . import surfaces as S
from .assess import assess_room
from .io_video import Keyframes
from .points_video import fit_ratio
from .schema import RoomPlan
from .sfm import SfmResult

MAX_VIEWS = 10  # frames per surface
MIN_GAIN = 0.02  # a further frame must add this fraction of the best total view quality
LATTICE_STEP = 0.15  # m between the sample points that judge how well a frame sees a surface
ALIGN_STRIDE = 8  # pixels between the samples that put a frame's depth on a surface's plane
ALIGN_BAND = (0.6, 1.6)  # model depth over the plane's depth: only pixels inside this can be the surface itself
ALIGN_MIN_PIXELS = 300
KINDS = ("wall", "ceiling")  # the floor is not searched: the flats have glossy tiles whose reflections read as stains

Depth = Callable[[np.ndarray], np.ndarray]


def world_poses(sfm: SfmResult, rotation: np.ndarray, scale: float) -> np.ndarray:
    """(N, 4, 4) camera-to-world in the plan's frame (gravity-aligned, metres, OpenCV axes); nan where not registered."""
    out = np.full((len(sfm.registered), 4, 4), np.nan)
    for i in np.nonzero(sfm.registered)[0]:
        out[i] = np.eye(4)
        out[i, :3, :3] = rotation @ sfm.cam_to_world[i]
        out[i, :3, 3] = rotation @ sfm.centers[i] * scale
    return out


def _lattice(plane: S.Plane) -> np.ndarray:
    """Sample points spread over a surface (inside its outline), to judge how well a frame sees it."""
    ns, nh = max(2, int(plane.size[0] / LATTICE_STEP)), max(2, int(plane.size[1] / LATTICE_STEP))
    s, h = np.meshgrid(np.linspace(0.03, plane.size[0] - 0.03, ns), np.linspace(0.03, plane.size[1] - 0.03, nh))
    if plane.inside is not None:
        rows, cols = plane.inside.shape
        keep = plane.inside[np.clip((h / S.MPP).astype(int), 0, rows - 1), np.clip((s / S.MPP).astype(int), 0, cols - 1)]
        s, h = s[keep], h[keep]
    return plane.origin[None, :] + s.reshape(-1, 1) * plane.u[None, :] + h.reshape(-1, 1) * plane.v[None, :]


@dataclass
class Views:
    """The clip's keyframes as ``surfaces.Frame`` objects, loaded on demand and only the ones a surface needs."""

    kf: Keyframes
    sfm: SfmResult
    poses: np.ndarray
    scale: float
    depth: Depth | None = None
    cache_size: int = 48
    align_depth: bool = True
    _frames: dict[int, S.Frame] = field(default_factory=dict)

    def _rgb(self, i: int) -> np.ndarray:
        bgr = cv2.imread(str(self.kf.directory / self.kf.names[i]))
        if bgr is None:
            raise ValueError(f"keyframe {self.kf.names[i]} cannot be read")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_m(self, i: int, rgb: np.ndarray) -> np.ndarray | None:
        """Depth model output for one keyframe in metres: fitted to that frame's SfM points, then scaled to metres."""
        if self.depth is None or i not in self.sfm.observations:
            return None
        uv, z = self.sfm.observations[i]
        d = np.asarray(self.depth(rgb), dtype=np.float32)
        r = fit_ratio(d, uv, z) if len(z) else None
        if r is None or r <= 0:
            return None
        return d / r * self.scale

    def frame(self, i: int) -> S.Frame:
        if i not in self._frames:
            rgb = self._rgb(i)
            fx, fy, cx, cy = self.sfm.intrinsics
            depth = self._depth_m(i, rgb)
            if len(self._frames) >= self.cache_size:
                self._frames.pop(next(iter(self._frames)))
            self._frames[i] = S.Frame(
                rgb=rgb,
                K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                pose=self.poses[i],
                depth=depth,
                depth_ok=None if depth is None else np.isfinite(depth),
            )
        return self._frames[i]

    def choose(self, plane: S.Plane, k: int = MAX_VIEWS) -> list[int]:
        """Keyframes that together see ``plane`` best: greedy on the gain in per-point view quality."""
        pts = _lattice(plane)
        idx = np.nonzero(self.sfm.registered)[0]
        fx, fy, cx, cy = self.sfm.intrinsics
        w_img, h_img = self.sfm.image_size
        quality = np.zeros((len(idx), len(pts)))
        for row, i in enumerate(idx):
            R, t = self.poses[i][:3, :3], self.poses[i][:3, 3]
            x = (pts - t) @ R
            z = x[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = fx * x[:, 0] / z + cx, fy * x[:, 1] / z + cy
            ray = pts - t
            cos = np.abs(ray @ plane.normal) / np.maximum(np.linalg.norm(ray, axis=1), 1e-6)
            ok = (z > S.Z_MIN) & (z < S.Z_MAX) & (cos > S.COS_MIN) & (u >= 1) & (u < w_img - 2) & (v >= 1) & (v < h_img - 2)
            quality[row] = np.where(ok, cos**3 / np.maximum(z, 0.5) ** 2, 0.0)
        total = quality.max(axis=0).sum()
        best = np.zeros(len(pts))
        chosen: list[int] = []
        for _ in range(k):
            gain = np.maximum(quality - best, 0).sum(axis=1)
            j = int(gain.argmax())
            if total <= 0 or gain[j] < MIN_GAIN * total:
                break
            chosen.append(int(idx[j]))
            best = np.maximum(best, quality[j])
        return sorted(chosen)

    def _aligned(self, fr: S.Frame, plane: S.Plane) -> S.Frame:
        """``fr`` with its depth rescaled so that the pixels that fall on ``plane`` read the plane's own depth.

        The depth model's scale is fitted to the sparse SfM points of the whole frame and is off by a few percent
        on any one wall (and a wall of the plan can sit several centimetres off). Left as it is, that is more than
        the gate's tolerance and the wall itself is thrown out; after this only what really stands in front of it
        or behind it disagrees. Frames with too few pixels on the plane keep their depth.
        """
        if fr.depth is None:
            return fr
        h, w = fr.depth.shape
        v, u = np.mgrid[ALIGN_STRIDE // 2 : h : ALIGN_STRIDE, ALIGN_STRIDE // 2 : w : ALIGN_STRIDE]
        sx, sy = fr.rgb.shape[1] / w, fr.rgb.shape[0] / h
        ray = np.stack([(u * sx - fr.K[0, 2]) / fr.K[0, 0], (v * sy - fr.K[1, 2]) / fr.K[1, 1], np.ones(u.shape)], axis=-1) @ fr.pose[:3, :3].T
        c = fr.pose[:3, 3]
        denom = ray @ plane.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane.origin - c) @ plane.normal / denom  # depth along the optical axis, since the ray has z = 1
        hit = c + t[..., None] * ray - plane.origin
        s_, h_ = hit @ plane.u, hit @ plane.v
        d = fr.depth[v, u]
        on = (t > S.Z_MIN) & (t < S.Z_MAX) & (s_ >= 0) & (s_ <= plane.size[0]) & (h_ >= 0) & (h_ <= plane.size[1]) & (d > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            near = on & (d / t > ALIGN_BAND[0]) & (d / t < ALIGN_BAND[1])
        if near.sum() < ALIGN_MIN_PIXELS:
            return fr
        r = float(np.median(t[near] / d[near]))
        return S.Frame(rgb=fr.rgb, K=fr.K, pose=fr.pose, depth=fr.depth * r, depth_ok=fr.depth_ok, undo=fr.undo)

    def frames_for(self, plane: S.Plane) -> list[S.Frame]:
        return [self._aligned(self.frame(i), plane) if self.align_depth else self.frame(i) for i in self.choose(plane)]


def assess_video(
    room: RoomPlan,
    floor_y: float,
    kf: Keyframes,
    sfm: SfmResult,
    rotation: np.ndarray,
    scale: float,
    scale_rel_sigma: float,
    depth: Depth | None = None,
    others: Sequence[RoomPlan] = (),
    debug_dir: Path | None = None,
) -> list[str]:
    """Fill ``room.damage``, ``room.concealed_flags`` and ``room.scope`` for one clip's room; returns notes.

    ``rotation`` and ``scale`` are the ones that took the SfM reconstruction into the plan's frame, ``floor_y`` is
    the floor height in that frame and ``scale_rel_sigma`` the scale's relative standard error. ``depth`` is the
    depth model (RGB in, z-depth out): with it, what stands in front of a surface is left out of its patch.
    ``debug_dir`` gets one picture per surface with its regions outlined.
    """
    views = Views(kf, sfm, world_poses(sfm, rotation, scale), scale, depth)
    ceiling_y = None if room.ceiling_height.value is None else floor_y + room.ceiling_height.value
    return assess_room(
        room, floor_y, ceiling_y, views.frames_for, others, KINDS, use_relief=False, scale_rel_sigma=scale_rel_sigma,
        debug_dir=debug_dir,
    )


def renumber_damage(room: RoomPlan, k: int) -> None:
    """Move the damage of a room that was judged as ``room_0`` with ``r0_`` walls to ``room_<k>`` and ``r<k>_``.

    Every clip is judged on its own, so its ids start at zero; when clips become rooms of one property the ids of
    the walls and of the room change (``video_pipeline._renumber``) and everything that names them must follow:
    the surface of each region, flag and scope item, and the ids of the regions, flags and items themselves.
    Call it after the room's own id and walls were renumbered, or before: it only looks at the old prefixes.
    """
    def fix(text: str) -> str:
        if text.startswith("room_0"):
            text = f"room_{k}" + text[len("room_0"):]
        return text.replace("r0_", f"r{k}_", 1) if text.startswith("r0_") else text

    for item in [*room.damage, *room.concealed_flags, *room.scope]:
        item.id = fix(item.id)
        item.surface.id = fix(item.surface.id)
        if hasattr(item, "damage_ids"):
            item.damage_ids = [fix(i) for i in item.damage_ids]
