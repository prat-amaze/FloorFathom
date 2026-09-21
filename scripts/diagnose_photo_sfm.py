"""Diagnose why the photo tier's SfM under-registers a room: shows the geometrically-verified
match graph (inliers per image pair after RANSAC), its connected components, and the dense-cloud
coverage. This distinguishes a capture problem (weak overlap between stills) from a parameter problem.
"""
import sys
import tempfile
import pathlib

import numpy as np
import pillow_heif
import pycolmap

pillow_heif.register_heif_opener()

from floorfathom.io_photos import load_photo_set, discover_rooms
from floorfathom.photo_pose import run_photo_sfm


def match_graph(folder: str):
    rooms = discover_rooms(pathlib.Path(folder))
    name = folder.rstrip("/").split("/")[-1]
    photos = load_photo_set(name, list(rooms.values())[0])
    n = len(photos.images)
    print(f"\n=== {folder}: {n} images ===")

    work = pathlib.Path(tempfile.mkdtemp())
    sfm = run_photo_sfm(photos, work, seed=0)

    # Re-open the database that run_photo_sfm built to read verified matches
    db = pycolmap.Database(str(work / "sfm" / "db.db"))
    images = {img.image_id: img.name for img in db.read_all_images()}
    id_of = {name: iid for iid, name in images.items()}
    short = {iid: images[iid].replace(".HEIC.jpg", "").replace("IMG_", "") for iid in images}

    # Verified (inlier) matches per pair
    print("\nGeometrically-verified matches (inliers) per pair:")
    pairs = {}
    for img_a in db.read_all_images():
        for img_b in db.read_all_images():
            if img_a.image_id < img_b.image_id:
                tvg = db.read_two_view_geometry(img_a.image_id, img_b.image_id)
                ninl = len(tvg.inlier_matches) if tvg is not None else 0
                if ninl > 0:
                    pairs[(img_a.image_id, img_b.image_id)] = ninl

    # Consecutive pairs (the walk order)
    ids_sorted = sorted(images, key=lambda i: images[i])
    print("  Consecutive (walk order):")
    for a, b in zip(ids_sorted, ids_sorted[1:]):
        key = (min(a, b), max(a, b))
        n_inl = pairs.get(key, 0)
        bar = "#" * (n_inl // 10)
        flag = "  <-- WEAK LINK" if n_inl < 30 else ""
        print(f"    {short[a]:>4} - {short[b]:>4}: {n_inl:4d} {bar}{flag}")

    # Connected components using a 30-inlier threshold (pycolmap's rough registration floor)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    idx = {iid: k for k, iid in enumerate(ids_sorted)}
    rows, cols = [], []
    for (a, b), ninl in pairs.items():
        if ninl >= 30:
            rows += [idx[a], idx[b]]
            cols += [idx[b], idx[a]]
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    print(f"\n  Connected components (>=30 inliers): {ncomp}")
    for c in range(ncomp):
        members = [short[ids_sorted[k]] for k in range(n) if labels[k] == c]
        print(f"    component {c}: {members}")

    if sfm is not None:
        reg = [short[id_of[nm]] for nm in [im.name if im.name.endswith('.jpg') else im.name + '.jpg' for im in photos.images] if False]
        regd = [str(photos.images[i].name).replace("IMG_", "").replace(".HEIC", "")
                for i in range(n) if sfm.registered[i]]
        print(f"\n  SfM registered {sfm.registered.sum()}/{n}: {regd}")
        print(f"  n_models={sfm.n_models}, flags={sfm.flags}")
    else:
        print("\n  SfM returned None")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["Data/Hall", "Data/Hall2"]:
        match_graph(f)
