from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_emotions.io_utils import ensure_parent


ACTIVATION_CACHE_VERSION = 1


def parse_layer_indices(raw_value: str) -> list[int]:
    layer_indices = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer_indices.append(int(chunk))
    if not layer_indices:
        raise ValueError("At least one layer index is required.")
    return list(dict.fromkeys(layer_indices))


def write_cache_manifest(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_cache_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_row_filename(index: int) -> str:
    return f"{index:06d}.pt"


def write_cache_row(path: Path, payload: dict) -> None:
    ensure_parent(path)
    torch.save(payload, path)


def load_cache_rows(cache_dir: Path, split: str) -> list[dict]:
    split_dir = cache_dir / split
    rows = []
    for path in sorted(split_dir.glob("*.pt")):
        rows.append(torch.load(path, map_location="cpu"))
    return rows
