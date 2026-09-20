"""Monocular metric depth from one RGB image. Any tier can use it; it knows nothing about photos.

The model is Depth Anything V2 Metric-Indoor Small (see ``models.py``). Its scale depends on how
the image is framed (a square crop reads about 15% nearer than the full frame), so the caller
sets the input size and this module does no cropping of its own. torch and transformers are
imported inside the constructor so runs that never build an estimator do not pay for them.
"""

from __future__ import annotations

import numpy as np

from .models import DEPTH, resolve

PATCH = 14  # the backbone's patch size: input sides must be multiples of it


def input_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """(height, width) fed to the model: native aspect, long side near ``long_side``, multiples of 14."""
    s = long_side / max(h, w)
    return max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH)


class DepthEstimator:
    """``depth = DepthEstimator()(rgb)``: float32 z-depth in metres, same height and width as ``rgb``."""

    def __init__(self, threads: int = 1):
        import torch
        from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

        torch.set_num_threads(threads)
        folder = resolve(DEPTH)
        self._processor = DPTImageProcessorPil.from_pretrained(folder)
        self._net = AutoModelForDepthEstimation.from_pretrained(folder, local_files_only=True).eval()

    def __call__(self, rgb: np.ndarray, long_side: int = 924) -> np.ndarray:
        import torch
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        h, w = rgb.shape[:2]
        nh, nw = input_size(h, w, long_side)
        image = Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC)
        with torch.no_grad():
            pixels = self._processor(images=image, do_resize=False, return_tensors="pt")["pixel_values"]
            depth = self._net(pixel_values=pixels).predicted_depth[None]  # (1, 1, nh, nw), metres
            depth = torch.nn.functional.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth[0, 0].numpy().astype(np.float32)
