"""Photo input: room discovery, EXIF orientation applied once, per-image focal length, honest flags."""

import numpy as np
from PIL import Image

from floorfathom.io_photos import DEFAULT_F35, discover_rooms, focal_px, load_photo_set


def _jpeg(path, size=(400, 300), f35=22, orientation=None, fill=(120, 60, 200)):
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    if f35:
        exif[0x8769] = {0xA405: f35}
    Image.new("RGB", size, fill).save(path, exif=exif)
    return path


def test_focal_length_from_the_35mm_equivalent_on_the_diagonal():
    assert abs(focal_px(22, 3024, 4032) - 2562.7) < 0.5  # the iPhone 16 ultra-wide at 0.8x
    assert abs(focal_px(22, 756, 1008) - 2562.7 / 4) < 0.2  # scales with the working size
    assert focal_px(17, 3024, 4032) < focal_px(22, 3024, 4032)  # Hall/IMG_4743 is wider


def test_exif_rotation_is_applied_exactly_once(tmp_path):
    p = _jpeg(tmp_path / "a.jpg", size=(400, 300), orientation=6)  # stored landscape, must be shown portrait
    im = load_photo_set("r", [p], long_side=200).images[0]
    assert im.rgb.shape == (200, 150, 3) and im.full_size == (300, 400)
    q = _jpeg(tmp_path / "b.jpg", size=(400, 300), orientation=1)
    assert load_photo_set("r", [q], long_side=200).images[0].rgb.shape == (150, 200, 3)


def test_each_image_keeps_its_own_focal_length_and_the_room_is_flagged(tmp_path):
    a, b = _jpeg(tmp_path / "a.jpg", f35=22), _jpeg(tmp_path / "b.jpg", f35=17)
    s = load_photo_set("r", [a, b], long_side=200)
    assert [im.f35 for im in s.images] == [22.0, 17.0]
    assert s.images[0].f_px > s.images[1].f_px
    assert "focal_varies_within_room" in s.flags


def test_missing_focal_length_is_defaulted_and_flagged(tmp_path):
    s = load_photo_set("r", [_jpeg(tmp_path / "a.jpg", f35=None)], long_side=200)
    assert s.images[0].f35 is None and "focal_prior_default:a.jpg" in s.flags
    assert abs(s.images[0].f_px - focal_px(DEFAULT_F35, 200, 150)) < 1e-6


def test_unreadable_file_is_skipped_and_flagged_not_fatal(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    s = load_photo_set("r", [bad, _jpeg(tmp_path / "ok.jpg")], long_side=200)
    assert [im.name for im in s.images] == ["ok.jpg"] and "image_unreadable:bad.jpg" in s.flags


def test_identical_files_share_a_hash_and_different_ones_do_not(tmp_path):
    a, b = _jpeg(tmp_path / "a.jpg"), _jpeg(tmp_path / "b.jpg", fill=(1, 2, 3))
    s = load_photo_set("r", [a, a, b], long_side=200)
    assert s.images[0].sha256 == s.images[1].sha256 != s.images[2].sha256


def test_rooms_are_discovered_from_images_folder_bare_folders_or_a_single_folder(tmp_path):
    (tmp_path / "images" / "b2").mkdir(parents=True)
    (tmp_path / "images" / "b1").mkdir()
    (tmp_path / "images" / "empty").mkdir()
    _jpeg(tmp_path / "images" / "b2" / "2.jpg"), _jpeg(tmp_path / "images" / "b2" / "1.JPG")
    _jpeg(tmp_path / "images" / "b1" / "1.png")
    (tmp_path / "images" / "b1" / "notes.txt").write_text("x")
    found = discover_rooms(tmp_path)
    assert list(found) == ["b1", "b2"] and [p.name for p in found["b2"]] == ["1.JPG", "2.jpg"]

    bare = tmp_path / "bare"
    (bare / "kitchen").mkdir(parents=True)
    _jpeg(bare / "kitchen" / "1.jpg")
    assert list(discover_rooms(bare)) == ["kitchen"]

    single = tmp_path / "single"
    single.mkdir()
    _jpeg(single / "1.jpg")
    assert list(discover_rooms(single)) == ["single"]
