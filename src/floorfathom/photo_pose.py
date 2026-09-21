"""Relative camera rotations between the stills of one room, from matched image features.

The capture protocol turns in place, so there is no baseline to triangulate and SfM is
degenerate. Each pair of overlapping photos instead gives a pure rotation between their viewing
rays; the best-supported pairs form a spanning tree that chains every photo into the frame of
one root photo. Hand-held arm swing (0.1-0.4 m) is not modelled and shows up as pose error.
Deterministic: all randomness comes from a seeded numpy generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

from .io_photos import PhotoImage

MAX_FEATURES = 4000
MAX_MATCHES = 3000  # more than this per pair adds cost, not accuracy
RATIO = 0.8


@dataclass
class Poses:
    rotations: list[np.ndarray | None]  # per image: camera -> root camera; None if unregistered
    root: int | None
    pair_inliers: dict[tuple[int, int], int]
    loop_error_deg: float | None  # worst disagreement of a link with the chain through the tree
    flags: list[str] = field(default_factory=list)


def _features(im: PhotoImage):
    import cv2

    grey = cv2.cvtColor(im.rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)  # dim rooms have weak contrast
    kp, desc = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=0.02).detectAndCompute(grey, None)
    if desc is None or len(kp) < 3:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], float), desc


def _rays(pts: np.ndarray, im: PhotoImage) -> np.ndarray:
    h, w = im.rgb.shape[:2]
    r = np.c_[(pts[:, 0] - w / 2) / im.f_px, (pts[:, 1] - h / 2) / im.f_px, np.ones(len(pts))]
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _match(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Indices (i, j) of mutual nearest neighbours that pass the ratio test."""
    import cv2

    if len(da) < 3 or len(db) < 3:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(da, db, k=2)
    back = {m.queryIdx: m.trainIdx for m in (p[0] for p in bf.knnMatch(db, da, k=2) if len(p) == 2)}
    out = [(a.queryIdx, a.trainIdx) for a, b in (p for p in fwd if len(p) == 2)
           if a.distance < RATIO * b.distance and back.get(a.trainIdx) == a.queryIdx]
    return np.array(out, int).reshape(-1, 2)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |R a - b|^2 over the rows of a and b (also batched: (k, n, 3))."""
    h = np.einsum("...ni,...nj->...ij", a, b)
    u, _, vt = np.linalg.svd(h)
    v = np.swapaxes(vt, -1, -2)
    d = np.sign(np.linalg.det(v @ np.swapaxes(u, -1, -2)))
    fix = np.broadcast_to(np.eye(3), v.shape).copy()
    fix[..., 2, 2] = d
    return v @ fix @ np.swapaxes(u, -1, -2)


def rotation_ransac(a: np.ndarray, b: np.ndarray, rng, thresh_deg: float, iters: int = 400):
    """(R, inlier mask) with R a ~ b for most rows, or (None, empty mask)."""
    n = len(a)
    if n < 3:
        return None, np.zeros(n, bool)
    cos_t = np.cos(np.radians(thresh_deg))
    idx = rng.integers(0, n, (iters, 3))
    cand = kabsch(a[idx], b[idx])
    counts = ((np.einsum("kij,nj->kni", cand, a) * b).sum(-1) > cos_t).sum(1)
    r = cand[int(np.argmax(counts))]
    for _ in range(2):  # refit on all inliers, recount
        mask = (a @ r.T * b).sum(-1) > cos_t
        if mask.sum() < 3:
            return None, np.zeros(n, bool)
        r = kabsch(a[mask], b[mask])
    return r, (a @ r.T * b).sum(-1) > cos_t


def _angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def register_rotations(images: list[PhotoImage], seed: int = 0, min_inliers: int = 20,
                       thresh_deg: float = 1.5) -> Poses:
    n = len(images)
    rng = np.random.default_rng(seed)
    feats = [_features(im) for im in images]
    links: dict[tuple[int, int], tuple[np.ndarray, int]] = {}  # (i, j), i < j -> (R i->j, inliers)
    for i in range(n):
        for j in range(i + 1, n):
            m = _match(feats[i][1], feats[j][1])[:MAX_MATCHES]
            if len(m) < min_inliers:
                continue
            a, b = _rays(feats[i][0][m[:, 0]], images[i]), _rays(feats[j][0][m[:, 1]], images[j])
            r, mask = rotation_ransac(a, b, rng, thresh_deg)
            if r is not None and mask.sum() >= min_inliers:
                links[(i, j)] = (r, int(mask.sum()))

    flags: list[str] = []
    rotations: list[np.ndarray | None] = [None] * n
    inliers = {k: v[1] for k, v in links.items()}
    if not links:
        flags += [f"image_unregistered:{im.name}" for im in images] + ["insufficient_registration"]
        return Poses(rotations, None, inliers, None, flags)

    ii, jj = zip(*links)
    w = np.array([1.0 / links[k][1] for k in links])  # strong links are cheap, so the tree keeps them
    graph = coo_matrix((w, (ii, jj)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    best = int(np.argmax(sizes))  # ties go to the lowest label, so the choice is deterministic
    members = [i for i in range(n) if labels[i] == best]
    strength = {i: sum(v[1] for k, v in links.items() if i in k) for i in members}
    root = max(members, key=lambda i: (strength[i], -i))

    tree = minimum_spanning_tree(graph).tocsr()
    tree = tree + tree.T
    order, parent = breadth_first_order(tree, root, directed=False)
    rotations[root] = np.eye(3)
    for c in order[1:]:
        p = int(parent[c])
        r_pc = links[(p, c)][0] if (p, c) in links else links[(c, p)][0].T  # p -> c
        rotations[c] = rotations[p] @ r_pc.T  # c -> p -> root

    worst = 0.0
    for (i, j), (r, _) in links.items():
        if rotations[i] is not None and rotations[j] is not None:
            worst = max(worst, _angle_deg(rotations[j].T @ rotations[i] @ r.T))  # chain vs measured i -> j
    if worst > 3.0:
        flags.append("pose_loop_inconsistent")
    flags += [f"image_unregistered:{images[i].name}" for i in range(n) if rotations[i] is None]
    if sum(r is not None for r in rotations) < 2:
        flags.append("insufficient_registration")
    return Poses(rotations, root, inliers, worst, flags)


DEFAULT_F35 = 24.0  # iPhone main camera, when EXIF focal is missing


def run_photo_sfm(photos: "PhotoSet", work, seed: int = 0) -> "SfmResult | None":
    """Reconstruct one room's stills with pycolmap, PER_IMAGE camera mode. Returns None if no model builds."""
    import cv2
    import numpy as np
    import pycolmap

    from .io_photos import PhotoSet
    from .sfm import SfmResult

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    images_dir = work / "images"
    images_dir.mkdir(exist_ok=True)
    db, sparse = work / "db.db", work / "sparse"
    if db.exists():
        db.unlink()
    if sparse.exists():
        import shutil
        shutil.rmtree(sparse)
    sparse.mkdir(exist_ok=True)

    names = []
    used_default_focal = False
    f_pxs = []
    for im in photos.images:
        name = im.name if Path(im.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else im.name + ".jpg"
        p = images_dir / name
        cv2.imwrite(str(p), cv2.cvtColor(im.rgb, cv2.COLOR_RGB2BGR))
        names.append(name)
        if im.f35 is None:
            used_default_focal = True
        f_pxs.append(im.f_px)

    w, h = photos.images[0].rgb.shape[1], photos.images[0].rgb.shape[0]
    f_prior = float(np.median(f_pxs))

    pycolmap.set_random_seed(seed)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{f_prior},{w / 2},{h / 2},0"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = 0.004
    extraction.sift.max_num_features = 8192
    cpu = pycolmap.Device.cpu
    pycolmap.extract_features(
        db, images_dir, image_names=names, camera_mode=pycolmap.CameraMode.PER_IMAGE,
        reader_options=reader, extraction_options=extraction, device=cpu,
    )
    # Sequential matching (each image to its neighbours) suits a walking capture; exhaustive is added
    # on top so that any two overlapping images can still form an initial pair.
    n_imgs = len(names)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = max(4, n_imgs - 1)  # large window ≈ exhaustive for small rooms
    pairing.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pairing, device=cpu)
    if n_imgs <= 12:  # add exhaustive on top for small sets: catches non-adjacent overlap
        pycolmap.match_exhaustive(db, device=cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.ba_refine_extra_params = False
    opts.min_model_size = 3  # keep partial models; default 10 discards small indoor reconstructions
    opts.mapper.abs_pose_min_num_inliers = 15  # default 30; fewer needed for close-range indoor
    opts.mapper.random_seed = seed  # deterministic mapping
    recs = pycolmap.incremental_mapping(db, images_dir, sparse, opts)
    if not recs:
        return None

    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    n = len(photos.images)
    index = {name: i for i, name in enumerate(names)}
    registered = np.zeros(n, dtype=bool)
    centers = np.full((n, 3), np.nan)
    rot = np.full((n, 3, 3), np.nan)
    observations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for image_id in rec.reg_image_ids():
        img = rec.images[image_id]
        i = index[img.name]
        cam_from_world = img.cam_from_world()
        world_from_cam = cam_from_world.inverse()
        registered[i] = True
        centers[i] = world_from_cam.translation
        rot[i] = world_from_cam.rotation.matrix()
        r_cw = cam_from_world.rotation.matrix()
        t_cw = np.asarray(cam_from_world.translation)
        uv, z = [], []
        for p in img.points2D:
            if p.has_point3D():
                depth = float((r_cw @ rec.points3D[p.point3D_id].xyz + t_cw)[2])
                if depth > 0:
                    uv.append(p.xy)
                    z.append(depth)
        observations[i] = (np.array(uv).reshape(-1, 2), np.array(z))

    cam0 = next(iter(rec.cameras.values()))
    pts = list(rec.points3D.values())
    result = SfmResult(
        registered=registered, centers=centers, cam_to_world=rot,
        intrinsics=(float(cam0.params[0]), float(cam0.params[0]), float(cam0.params[1]), float(cam0.params[2])),
        image_size=(w, h),
        points=np.array([p.xyz for p in pts]).reshape(-1, 3) if pts else np.zeros((0, 3), float),
        track_length=np.array([p.track.length() for p in pts], dtype=int) if pts else np.zeros(0, int),
        point_error=np.array([p.error for p in pts], dtype=float) if pts else np.zeros(0, float),
        n_models=len(recs), mean_reprojection_px=float(rec.compute_mean_reprojection_error()),
        observations=observations,
    )
    if used_default_focal:
        result.flags.append("focal_prior_default")
    if len(recs) > 1:
        result.flags.append("sfm_split_into_several_models")
    if result.registered_fraction < 0.6:
        result.flags.append("sfm_registered_too_few_frames")
    return result
