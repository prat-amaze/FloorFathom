"""Photo capture input: one folder per room, stills with EXIF (iPhone HEIC, or JPEG/PNG).

Each ``images/<room>/`` folder is one room. Every image carries its own focal length, because
the zoom can differ inside one folder (Hall/IMG_4743 is 17 mm-equivalent, the rest 22 mm), so
the camera matrix is never shared. Only the EXIF tags needed here are read; GPS is not kept.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
DIAG_35MM = 43.267  # diagonal of a 36 x 24 mm frame; the 35 mm-equivalent focal is defined on it
DEFAULT_F35 = 24.0  # main iPhone camera, used only when EXIF has no focal length (flagged)
FOCAL_REL_SIGMA = 0.03  # EXIF stores the equivalent as a rounded integer; lens distortion is unknown


@dataclass
class PhotoImage:
    name: str
    sha256: str  # of the file bytes: cache key for model output
    rgb: np.ndarray  # (H, W, 3) uint8, upright, long side = ``long_side``
    full_size: tuple[int, int]  # (width, height) of the decoded original
    f35: float | None  # EXIF 35 mm-equivalent focal length, None if absent
    f_px: float  # focal length in pixels of ``rgb``


@dataclass
class PhotoSet:
    room: str
    images: list[PhotoImage] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def focal_px(f35: float, width: int, height: int) -> float:
    """Focal length in pixels for an image of this size: f35 * diagonal / 43.267 (pinhole, no distortion)."""
    return f35 * float(np.hypot(width, height)) / DIAG_35MM


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def discover_rooms(capture: Path) -> dict[str, list[Path]]:
    """Room name -> image paths (sorted). ``images/<room>/``, else ``<room>/``, else the folder itself is one room."""
    base = capture / "images" if (capture / "images").is_dir() else capture
    rooms = {d.name: _images_in(d) for d in sorted(base.iterdir()) if d.is_dir()}
    rooms = {name: paths for name, paths in rooms.items() if paths}
    if not rooms and _images_in(base):
        rooms = {base.name: _images_in(base)}
    return rooms


def _read(path: Path, long_side: int) -> PhotoImage:
    from PIL import Image, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        exif = im.getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        im = ImageOps.exif_transpose(im)  # applied once; a no-op for HEIC, whose decoder already rotated
        full = im.size
        scale = long_side / max(full)
        rgb = np.asarray(im.convert("RGB").resize((round(full[0] * scale), round(full[1] * scale)), Image.BICUBIC))
    f = float(f35) if f35 else None
    return PhotoImage(path.name, hashlib.sha256(path.read_bytes()).hexdigest(), rgb, full, f,
                      focal_px(f or DEFAULT_F35, rgb.shape[1], rgb.shape[0]))


def load_photo_set(room: str, paths: list[Path], long_side: int = 1008) -> PhotoSet:
    """Load one room's stills. Unreadable files are skipped and flagged; a missing focal length is defaulted and flagged."""
    out = PhotoSet(room)
    for path in paths:
        try:
            out.images.append(_read(path, long_side))
        except Exception:  # a corrupt or unsupported file must not stop the rest of the capture
            out.flags.append(f"image_unreadable:{path.name}")
    for im in out.images:
        if im.f35 is None:
            out.flags.append(f"focal_prior_default:{im.name}")
    if len({im.f35 for im in out.images if im.f35}) > 1:
        out.flags.append("focal_varies_within_room")
    return out
