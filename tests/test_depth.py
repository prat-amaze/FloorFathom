"""DepthEstimator: model input sizing (pure) and the real model on a tiny image (skipped without weights)."""

import numpy as np
import pytest

from floorfathom.depth import PATCH, input_size
from floorfathom.models import DEPTH, ModelError, resolve


@pytest.mark.parametrize("h, w", [(1008, 756), (4032, 3024), (250, 333), (720, 1280), (5, 3000)])
def test_input_size_is_multiples_of_14_near_the_requested_long_side(h, w):
    nh, nw = input_size(h, w, 924)
    assert nh % PATCH == 0 and nw % PATCH == 0 and nh >= PATCH and nw >= PATCH
    assert abs(max(nh, nw) - 924) <= PATCH
    if min(h, w) > 100:
        assert abs(nh / nw - h / w) < 0.05 * h / w  # native aspect survives the rounding


def _weights_present() -> bool:
    try:
        resolve(DEPTH)
        return True
    except ModelError:
        return False


@pytest.mark.skipif(not _weights_present(), reason="run `floorfathom fetch-models` first")
def test_metric_depth_at_input_resolution_and_deterministic():
    from floorfathom.depth import DepthEstimator

    rgb = np.random.default_rng(0).integers(0, 256, (250, 333, 3), dtype=np.uint8)
    est = DepthEstimator()
    d = est(rgb, long_side=224)
    assert d.shape == (250, 333) and d.dtype == np.float32
    assert np.isfinite(d).all() and d.min() > 0 and d.max() <= 20  # metres, the model's indoor range
    assert np.array_equal(d, est(rgb, long_side=224))
    with pytest.raises(ValueError):
        est(rgb.astype(np.float32))
