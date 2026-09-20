"""Model registry: offline resolve fails loudly and helpfully. No network, no weights needed."""

import pytest

from floorfathom.models import DEPTH, REGISTRY, ModelError, resolve


def test_missing_model_says_how_to_fetch(tmp_path):
    with pytest.raises(ModelError, match="fetch-models"):
        resolve(DEPTH, root=tmp_path)


def test_unknown_model_lists_the_known_ones(tmp_path):
    with pytest.raises(ModelError, match=DEPTH):
        resolve("no-such-model", root=tmp_path)


def test_file_with_the_wrong_hash_is_rejected(tmp_path):
    folder = tmp_path / DEPTH
    folder.mkdir()
    for name in REGISTRY[DEPTH].files:
        (folder / name).write_bytes(b"not the real weights")
    with pytest.raises(ModelError, match="pinned hash"):
        resolve(DEPTH, root=tmp_path)
