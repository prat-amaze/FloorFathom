"""Pretrained models: download once with ``floorfathom fetch-models``, load offline after.

Weights are not committed (see .gitignore). Every model is pinned to one revision and every
file to a SHA-256, so a run is reproducible and a corrupt or swapped file is caught. Only
``fetch`` touches the network; ``resolve`` never does. Nothing here knows what a model is
used for: a tier that needs a new model adds a ``ModelSpec`` and calls ``resolve``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class ModelError(RuntimeError):
    """A model is missing, or its files do not match the pinned hashes."""


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    licence: str
    files: dict[str, str]  # file name -> sha256


DEPTH = "depth-anything-v2-metric-indoor-small"

REGISTRY: dict[str, ModelSpec] = {
    DEPTH: ModelSpec(
        repo_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        revision="8078d68a9c75a972131914f6afd0c1723be0da7f",
        licence="Apache-2.0 per the upstream Depth-Anything-V2 repository; the -hf conversion declares none",
        files={
            "config.json": "0b1d9fc591693d16b864249a9cfcf5264c8c00be999c1643fa1c64dc105d55f8",
            "preprocessor_config.json": "533b16a60445d7cab5086d39b45f92be45624b977972c62d5984d93e98366063",
            "model.safetensors": "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3",
        },
    ),
}


def models_dir() -> Path:
    """``FLOORFATHOM_MODELS`` if set, else ``models/`` in the repo, else a per-user cache."""
    env = os.environ.get("FLOORFATHOM_MODELS")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "models"
    return Path.home() / ".cache" / "floorfathom" / "models"


def _spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ModelError(f"unknown model {name!r}; known: {', '.join(REGISTRY)}") from None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _problems(name: str, folder: Path) -> list[str]:
    out = []
    for fname, want in _spec(name).files.items():
        p = folder / fname
        if not p.is_file():
            out.append(f"{fname} is missing")
        elif _sha256(p) != want:
            out.append(f"{fname} does not match the pinned hash")
    return out


def resolve(name: str, root: Path | None = None) -> Path:
    """Local folder of a fetched model. Never touches the network."""
    folder = (root or models_dir()) / name
    problems = _problems(name, folder)
    if problems:
        raise ModelError(f"model {name!r} is not usable in {folder} ({'; '.join(problems)}). "
                         f"Run: floorfathom fetch-models --model {name}")
    return folder


def fetch(name: str, root: Path | None = None) -> Path:
    """Download ``name`` at its pinned revision, then verify it. A verified copy is left alone."""
    spec = _spec(name)
    folder = (root or models_dir()) / name
    if not _problems(name, folder):
        return folder
    from huggingface_hub import snapshot_download

    snapshot_download(spec.repo_id, revision=spec.revision, local_dir=folder, allow_patterns=list(spec.files))
    problems = _problems(name, folder)
    if problems:
        raise ModelError(f"downloaded {name!r} but it failed verification: {'; '.join(problems)}")
    return folder
