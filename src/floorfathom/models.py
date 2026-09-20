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
CLIP = "clip-vit-base-patch32"

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
    CLIP: ModelSpec(
        repo_id="openai/clip-vit-base-patch32",
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        licence="MIT (OpenAI CLIP)",
        files={
            "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
            "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
            "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
            "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
            "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
            "tokenizer_config.json": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
            "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
            "pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
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
