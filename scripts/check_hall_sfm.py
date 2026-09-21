import pillow_heif, tempfile, pathlib, numpy as np
pillow_heif.register_heif_opener()
from floorfathom.io_photos import load_photo_set, discover_rooms
from floorfathom.photo_pose import run_photo_sfm

for folder in ["Data/Hall", "Data/Hall2"]:
    rooms = discover_rooms(pathlib.Path(folder))
    photos = load_photo_set(folder.split("/")[-1], list(rooms.values())[0])
    work = pathlib.Path(tempfile.mkdtemp())
    sfm = run_photo_sfm(photos, work, seed=0)
    if sfm is None:
        print(f"{folder}: SfM=None -> rotation-only fallback")
    else:
        cs = sfm.centers[sfm.registered]
        spread = cs.max(axis=0) - cs.min(axis=0) if sfm.registered.sum() >= 2 else np.zeros(3)
        print(f"{folder}: {sfm.registered.sum()}/{len(photos.images)} registered, "
              f"spread_max={spread.max():.3f} SfM-units, flags={sfm.flags}")
