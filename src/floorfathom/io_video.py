"""Loader for video-tier captures: a .MOV becomes a folder of sharp, evenly spaced keyframes.

Structure-from-Motion needs frames that share visible content, and motion blur destroys the
features it matches. So the clip is cut into short windows and the sharpest frame (variance
of the Laplacian) of each window is kept. Every keyframe remembers which video frame and
which second it came from, so a broken reconstruction can be pointed at a moment in the
clip and the uncertainty step can leave out contiguous stretches of time.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframes:
    directory: Path
    names: list[str]  # file names inside ``directory``, in time order
    source_index: np.ndarray  # (N,) video frame number each keyframe was taken from
    time_s: np.ndarray  # (N,) seconds from the start of the clip
    sharpness: np.ndarray  # (N,) variance of the Laplacian at the saved resolution
    size: tuple[int, int]  # (width, height) of the saved keyframes
    fps: float
    video: Path

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class CameraMetadata:
    """What the phone wrote into the clip about its lens. Every field may be None."""

    focal_35mm_equivalent: float | None  # whole millimetres, rounded by the phone
    lens_model: str | None
    device_model: str | None


_LENS_KEYS = {
    "com.apple.quicktime.camera.focal_length.35mm_equivalent": "focal",
    "com.apple.quicktime.camera.lens_model": "lens",
    "com.apple.quicktime.model": "device",
}


def _boxes(f, start: int, end: int):
    """(type, body_start, end) of the QuickTime boxes lying between two file offsets."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        yield kind, pos + header, pos + size
        pos += size


def _decode(kind: int, raw: bytes):
    if kind == 1:
        return raw.decode("utf8", "replace")
    if kind == 23 and len(raw) == 4:
        return struct.unpack(">f", raw)[0]
    if kind == 24 and len(raw) == 8:
        return struct.unpack(">d", raw)[0]
    if kind in (21, 65, 66, 67):
        return int.from_bytes(raw, "big", signed=True)
    return None


def _meta_values(f, start: int, end: int) -> dict[str, object]:
    """Lens-related key/value pairs of one QuickTime ``meta`` box (GPS and the rest are skipped)."""
    f.seek(start + 4)
    # a QuickTime meta box starts straight with its children; an ISO one has a 4 byte version first
    skip = 0 if f.read(4) in (b"hdlr", b"keys", b"ilst") else 4
    kids = {kind: (a, b) for kind, a, b in _boxes(f, start + skip, end)}
    if b"keys" not in kids or b"ilst" not in kids:
        return {}
    a, _ = kids[b"keys"]
    f.seek(a + 4)
    count = struct.unpack(">I", f.read(4))[0]
    names = []
    for _ in range(count):
        size = struct.unpack(">I", f.read(4))[0]
        f.read(4)  # key namespace
        names.append(f.read(size - 8).decode("latin1"))
    out: dict[str, object] = {}
    for item, a, b in _boxes(f, *kids[b"ilst"]):
        index = struct.unpack(">I", item)[0]
        if not 1 <= index <= len(names) or names[index - 1] not in _LENS_KEYS:
            continue
        for kind, x, y in _boxes(f, a, b):
            if kind == b"data":
                f.seek(x)
                type_code = struct.unpack(">I", f.read(4))[0] & 0xFFFFFF
                f.read(4)  # locale
                out[_LENS_KEYS[names[index - 1]]] = _decode(type_code, f.read(y - x - 8))
    return out


def read_camera_metadata(video: str | Path) -> CameraMetadata:
    """Lens information from the clip's QuickTime metadata. Never raises: it is only a prior."""
    found: dict[str, object] = {}
    try:
        with open(video, "rb") as f:
            f.seek(0, 2)
            top = list(_boxes(f, 0, f.tell()))
            for kind, s, e in top:
                if kind != b"moov":
                    continue
                for k2, s2, e2 in _boxes(f, s, e):
                    if k2 == b"meta":
                        found.update(_meta_values(f, s2, e2))
                    elif k2 == b"trak":
                        for k3, s3, e3 in _boxes(f, s2, e2):
                            if k3 == b"meta":
                                found.update(_meta_values(f, s3, e3))
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        pass
    try:  # the phone writes this number as text ("16")
        focal = float(found["focal"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        focal = None
    return CameraMetadata(
        focal_35mm_equivalent=focal if focal is not None and focal > 0 else None,
        lens_model=found.get("lens") if isinstance(found.get("lens"), str) else None,
        device_model=found.get("device") if isinstance(found.get("device"), str) else None,
    )


def extract_keyframes(video: str | Path, out_dir: str | Path, step_s: float = 0.15, width: int = 720) -> Keyframes:
    """Save the sharpest frame of every ``step_s`` seconds, resized to ``width`` pixels wide."""
    video, out_dir = Path(video), Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps > 0:
            raise ValueError(f"{video} reports no frame rate")
        window = max(1, round(step_s * fps))
        out_dir.mkdir(parents=True, exist_ok=True)

        names: list[str] = []
        src: list[int] = []
        sharp: list[float] = []
        size = (0, 0)
        best, best_score, best_i, i = None, -1.0, -1, 0
        while True:
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                score = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if score > best_score:
                    best, best_score, best_i = small, score, i
            i += 1
            if (i % window == 0 or not ok) and best is not None:
                name = f"{len(names):05d}.jpg"
                cv2.imwrite(str(out_dir / name), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
                names.append(name)
                src.append(best_i)
                sharp.append(best_score)
                size = (best.shape[1], best.shape[0])
                best, best_score = None, -1.0
            if not ok:
                break
    finally:
        cap.release()
    if not names:
        raise ValueError(f"no frames could be read from {video}")
    source_index = np.array(src)
    return Keyframes(out_dir, names, source_index, source_index / fps, np.array(sharp), size, fps, video)
