"""Debug artefacts: the raw geometry next to what was extracted from it.

  points.ply      the point cloud (y up), openable in MeshLab / CloudCompare
  topdown.png     the cloud from above with wall evidence, polygons and camera path
  height_hist.png height histogram with the floor and ceiling planes marked
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .estimate import Estimate  # noqa: E402
from .points import Cloud  # noqa: E402


def write_ply(path: Path, points: np.ndarray, max_points: int = 1_500_000) -> None:
    if len(points) > max_points:  # fixed stride keeps the file deterministic
        points = points[:: int(np.ceil(len(points) / max_points))]
    pts = np.ascontiguousarray(points, dtype="<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def write_debug(out: Path, cloud: Cloud, est: Estimate, scan) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = cloud.points
    write_ply(out / "points.ply", p)

    # top-down view, plan coordinates (x, -z)
    fig, ax = plt.subplots(figsize=(11, 9))
    hb = ax.hexbin(p[:, 0], -p[:, 2], gridsize=220, bins="log", mincnt=1, cmap="Greys", linewidths=0)
    if est.barrier is not None:
        g = est.grid
        rows, cols = np.nonzero(est.barrier)
        xy = g.to_xz(cols, rows)
        ax.plot(xy[:, 0], -xy[:, 1], ",", color="#b91c1c", alpha=0.6)
    for k, r in enumerate(est.rooms):
        poly = np.vstack([r.outline.polygon, r.outline.polygon[:1]])
        ax.plot(poly[:, 0], -poly[:, 1], "-", color="#1d4ed8", lw=2)
        c = r.outline.polygon.mean(axis=0)
        ax.text(c[0], -c[1], f"room_{k}", color="#1d4ed8", ha="center", fontsize=9)
    for a, b in est.closures:
        ax.plot([a[0], b[0]], [-a[1], -b[1]], "-", color="#c2410c", lw=2, alpha=0.7)
    ax.plot(scan.positions[:, 0], -scan.positions[:, 2], "-", color="#16a34a", lw=0.7, label="camera path")
    ax.set_aspect("equal")
    ax.set_title("top-down: points (grey), wall evidence (red), closures (orange), rooms (blue)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "topdown.png", dpi=110)
    plt.close(fig)

    # height histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    y = p[:, 1]
    ax.hist(y, bins=np.arange(y.min(), y.max(), 0.01), color="#6b7280")
    if est.floor is not None:
        ax.axvline(est.floor.y, color="#1d4ed8", label=f"floor {est.floor.y:.3f}")
    for k, r in enumerate(est.rooms):
        if r.ceiling is not None:
            ax.axvline(r.ceiling.y, color="#c2410c", alpha=0.5, label=f"room_{k} ceiling {r.ceiling.y:.3f}")
    ax.set_xlabel("world height y (m)")
    ax.set_ylabel("points per 1 cm")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "height_hist.png", dpi=110)
    plt.close(fig)
