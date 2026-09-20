"""Camera metadata of a clip and the starting focal length derived from it."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from floorfathom.io_video import Keyframes, read_camera_metadata
from floorfathom.sfm import DEFAULT_FOCAL_FRACTION, FOCAL_PER_MM, focal_prior


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def meta_box(items: dict[str, str]) -> bytes:
    """A QuickTime ``meta`` box: hdlr, keys and ilst, with every value stored as UTF-8 text."""
    keys = struct.pack(">II", 0, len(items))
    ilst = b""
    for i, (name, value) in enumerate(items.items(), start=1):
        keys += struct.pack(">I", 8 + len(name)) + b"mdta" + name.encode()
        data = struct.pack(">II", 1, 0) + value.encode()  # type 1 = UTF-8, locale 0
        ilst += struct.pack(">I", 8 + len(box(b"data", data))) + struct.pack(">I", i) + box(b"data", data)
    return box(b"meta", box(b"hdlr", b"\0" * 24) + box(b"keys", keys) + box(b"ilst", ilst))


def write_clip(path: Path, focal: str | None, with_gps: bool = True) -> Path:
    lens = {"com.apple.quicktime.camera.lens_model": "iPhone 16 back camera 2.22mm f/2.2"}
    if focal is not None:
        lens["com.apple.quicktime.camera.focal_length.35mm_equivalent"] = focal
    clip_level = {"com.apple.quicktime.model": "iPhone 16"}
    if with_gps:
        clip_level["com.apple.quicktime.location.ISO6709"] = "+12.9318+077.7452+885.972/"
    moov = box(b"moov", box(b"trak", box(b"tkhd", b"\0" * 84) + meta_box(lens)) + meta_box(clip_level))
    path.write_bytes(box(b"ftyp", b"qt  " + b"\0" * 8) + box(b"mdat", b"\0" * 64) + moov)
    return path


def keyframes(video: Path) -> Keyframes:
    z = np.zeros(1)
    return Keyframes(video.parent, ["0.jpg"], z, z, z, (720, 1280), 30.0, video)


def test_reads_lens_information(tmp_path):
    m = read_camera_metadata(write_clip(tmp_path / "a.MOV", "16"))
    assert m.focal_35mm_equivalent == 16.0
    assert m.lens_model == "iPhone 16 back camera 2.22mm f/2.2"
    assert m.device_model == "iPhone 16"


def test_clip_without_focal_or_gps_or_not_a_video_gives_none(tmp_path):
    assert read_camera_metadata(write_clip(tmp_path / "b.MOV", None, with_gps=False)).focal_35mm_equivalent is None
    not_video = tmp_path / "c.MOV"
    not_video.write_bytes(b"this is not a movie file at all")
    assert read_camera_metadata(not_video).focal_35mm_equivalent is None
    assert read_camera_metadata(tmp_path / "missing.MOV").focal_35mm_equivalent is None


def test_unreadable_focal_text_is_ignored(tmp_path):
    assert read_camera_metadata(write_clip(tmp_path / "d.MOV", "wide")).focal_35mm_equivalent is None
    assert read_camera_metadata(write_clip(tmp_path / "e.MOV", "-3")).focal_35mm_equivalent is None


def test_starting_focal_follows_the_zoom_of_the_clip(tmp_path):
    f16, src16 = focal_prior(keyframes(write_clip(tmp_path / "f.MOV", "16")))
    f14, src14 = focal_prior(keyframes(write_clip(tmp_path / "g.MOV", "14")))
    assert src16 == src14 == "metadata_35mm_equivalent"
    assert f16 == pytest.approx(16 * FOCAL_PER_MM * 1280)
    assert f14 / f16 == pytest.approx(14 / 16)  # a wider clip starts with a shorter focal length


def test_default_focal_used_and_labelled_when_metadata_is_missing(tmp_path):
    f, src = focal_prior(keyframes(write_clip(tmp_path / "h.MOV", None)))
    assert src == "default"
    assert f == pytest.approx(DEFAULT_FOCAL_FRACTION * 1280)
