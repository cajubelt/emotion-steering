from __future__ import annotations

from pathlib import Path

import torch


def load_vector_payload(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if "vectors" not in payload and "final_vectors" in payload:
        payload["vectors"] = payload["final_vectors"]
    if "vectors" not in payload:
        raise KeyError("Vector payload is missing a 'vectors' field.")
    if "layer_idx" not in payload:
        raise KeyError("Vector payload is missing a 'layer_idx' field.")
    payload.setdefault("format_version", 1)
    payload.setdefault("pooling_strategy", "mean")
    payload.setdefault("pool_size", 32)
    payload.setdefault("construction_mode", "grand_mean")
    payload.setdefault("pairwise_preset", None)
    return payload
