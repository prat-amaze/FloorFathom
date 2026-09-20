"""Rendered plan: one top-down drawing of every room, with dimensions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .schema import CapturePlan, RoomPlan  # noqa: E402

INK = "#1f2937"
MUTED = "#9ca3af"
FILL = "#eef4fb"
DOOR = "#c2410c"


def _fmt(m) -> str:
    if m.value is None:
        return "n/a"
    return f"{m.value:.2f} m"


def _label_outside(ax, p0, p1, text, centroid, colour=INK, size=7.5):
    mid = 0.5 * (np.array(p0) + np.array(p1))
    d = np.array(p1) - np.array(p0)
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-9)
    if np.dot(mid + 0.3 * n - centroid, mid - centroid) < 0:
        n = -n
    pos = mid + 0.28 * n
    ang = np.degrees(np.arctan2(d[1], d[0]))
    if ang > 90 or ang < -90:
        ang += 180
    ax.text(pos[0], pos[1], text, ha="center", va="center", fontsize=size, color=colour, rotation=ang, rotation_mode="anchor")


def draw_room(ax, room: RoomPlan, index: int) -> None:
    poly = np.array(room.polygon)
    ax.fill(poly[:, 0], poly[:, 1], color=FILL, zorder=1)
    centroid = poly.mean(axis=0)
    for w in room.walls:
        a, b = np.array(w.start), np.array(w.end)
        if w.evidence == "wall_points":
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=INK, lw=2.6, solid_capstyle="round", zorder=3)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=MUTED, lw=1.4, zorder=2)
        if w.length.value is not None and w.length.value >= 0.7:
            _label_outside(ax, a, b, _fmt(w.length), centroid)
    for o in room.openings:
        a, b = np.array(o.start), np.array(o.end)
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=DOOR, lw=4.0, alpha=0.85, solid_capstyle="butt", zorder=4)
    ch = "ceiling n/a" if room.ceiling_height.value is None else f"ceiling {room.ceiling_height.value:.2f} m"
    ax.text(
        centroid[0],
        centroid[1],
        f"{room.id.replace('_', ' ')}\n{room.floor_area.value:.1f} m$^2$\n{ch}",
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
        zorder=5,
    )


def render_plan(plan: CapturePlan, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    for i, room in enumerate(plan.rooms):
        draw_room(ax, room, i)
    ax.set_aspect("equal")
    ax.grid(True, color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("plan x (m)")
    ax.set_ylabel("plan y (m)")
    ax.set_title(f"{plan.capture}  -  {plan.tier} tier, {len(plan.rooms)} room(s)", fontsize=11, color=INK)
    handles = [
        plt.Line2D([0], [0], color=INK, lw=2.6, label="wall (fitted to lidar points)"),
        plt.Line2D([0], [0], color=MUTED, lw=1.4, ls="--", label="closure (no wall points)"),
        plt.Line2D([0], [0], color=DOOR, lw=4.0, label="doorway"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
