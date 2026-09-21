"""LiDAR-tier adapter for ``assess.assess_room``: sharp RGB frames of a scan with their depth, pose and drift undo.

The video is decoded once. In each window of the walk the sharpest of a few candidate frames is kept, downscaled
and written to a temporary folder, so every surface of every room can re-read the frames it needs cheaply.
Plan coordinates are drift-corrected; ``undo`` maps them back into the raw frame the poses were measured in.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .drift import Drift
from .io_lidar import LidarScan
from .schema import RoomPlan
from .surfaces import Frame, Plane, lattice_of, looks_at

TARGET_FRAMES = 240  # frames kept over the whole walk
CANDIDATES = 6  # frames looked at per window; the sharpest is kept
SCALE = 0.5  # cached picture size relative to the video
NEAR = 1.0  # m: frames from this far outside a room's outline are still used for it
PER_SURFACE = 70  # frames used for one surface at most


def make_undo(shift: np.ndarray, centre: np.ndarray):
    """The inverse of ``drift._apply`` for one chunk: corrected plan points back to the raw frame."""
    c, s = np.cos(shift[2]), np.sin(shift[2])

    def undo(points: np.ndarray) -> np.ndarray:
        rel = points[:, [0, 2]].astype(np.float64) - centre - shift[:2]
        out = points.copy()
        out[:, 0] = (c * rel[:, 0] + s * rel[:, 1] + centre[0]).astype(out.dtype)
        out[:, 2] = (-s * rel[:, 0] + c * rel[:, 1] + centre[1]).astype(out.dtype)
        return out

    return undo


def _sharpness(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


class LidarFrames:
    def __init__(self, scan: LidarScan, traj_plan: np.ndarray, drift: Drift | None, target: int = TARGET_FRAMES):
        """``traj_plan``: (N, 2) camera (x, z) of every scan row in the plan's (drift-corrected) frame."""
        self.scan, self.drift = scan, drift
        self.dir = Path(tempfile.mkdtemp(prefix="floorfathom_frames_"))
        self.rows = self._sample(target)
        self.plan_xy = np.stack([traj_plan[self.rows, 0], -traj_plan[self.rows, 1]], axis=1)
        w, h = scan.rgb_size
        dw, dh = scan.depth_size
        self._to_rgb = np.array([w / dw, h / dh, w / dw, h / dh]) * SCALE  # depth-pixel intrinsics to cached-picture pixels
        self._shape = (int(round(h * SCALE)), int(round(w * SCALE)))
        self._n = len(scan)

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _sample(self, target: int) -> np.ndarray:
        n = len(self.scan)
        edges = np.linspace(0, n, min(target, n) + 1).astype(int)
        cap = cv2.VideoCapture(str(self.scan.root / "rgb.mp4"))
        keep = []
        try:
            for a, b in zip(edges[:-1], edges[1:]):
                cands = set(np.unique(np.linspace(a, b - 1, CANDIDATES).astype(int)).tolist())
                best, best_img = -1.0, None
                for i in range(a, b):
                    if i in cands:
                        ok, img = cap.read()
                        if ok:
                            sh = _sharpness(img)
                            if sh > best:
                                best, best_img, best_row = sh, img, i
                    else:
                        cap.grab()
                if best_img is not None:
                    small = cv2.resize(best_img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(self.dir / f"{best_row:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    keep.append(best_row)
        finally:
            cap.release()
        return np.array(keep, dtype=int)

    def _pose(self, row: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(self.scan.quats[row]).as_matrix()
        pose[:3, 3] = self.scan.positions[row]
        return pose

    def _K(self, row: int) -> np.ndarray:
        fx, fy, cx, cy = self.scan.intrinsics[row] * self._to_rgb
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    def _undo(self, row: int):
        if self.drift is None:
            return None
        chunk = min(row * len(self.drift.shifts) // self._n, len(self.drift.shifts) - 1)
        return make_undo(self.drift.shifts[chunk], self.drift.centre)

    def _stub(self, row: int) -> Frame:
        return Frame(rgb=np.empty((*self._shape, 0), np.uint8), K=self._K(row), pose=self._pose(row), undo=self._undo(row))

    def load(self, row: int) -> Frame:
        img = cv2.imread(str(self.dir / f"{row:06d}.jpg"))
        depth = self.scan.depth_m(row)
        return Frame(
            rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            K=self._K(row),
            pose=self._pose(row),
            depth=depth,
            depth_ok=self.scan.confidence(row) >= 1,
            undo=self._undo(row),
        )

    def for_room(self, room: RoomPlan):
        """A function from a surface's plane to the frames that look at it, for ``assess_room``."""
        poly = np.array(room.polygon, float)
        lo, hi = poly.min(axis=0) - NEAR, poly.max(axis=0) + NEAR
        inside = np.nonzero(((self.plan_xy >= lo) & (self.plan_xy <= hi)).all(axis=1))[0]

        def frames(plane: Plane) -> Iterator[Frame]:
            lattice = lattice_of(plane)
            rows = [int(self.rows[k]) for k in inside if looks_at(lattice, self._stub(int(self.rows[k])))]
            if len(rows) > PER_SURFACE:
                rows = [rows[i] for i in np.linspace(0, len(rows) - 1, PER_SURFACE).astype(int)]
            for r in rows:
                yield self.load(r)

        return frames
