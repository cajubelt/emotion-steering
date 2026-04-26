#!/usr/bin/env python3
"""Stage 1 diagnostic: vector-geometry-only inspection of two runs.

Computes per-emotion L2 norm, largest/median norm ratio, and the pairwise
cosine-similarity matrix for each run, and prints markdown tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.vector_payloads import load_vector_payload


RUNS = {
    "vad_grid_8_e4b_mean50 (mean pooling, 'nervous' dominated)": Path(
        "vectors/vad_grid_8_e4b_mean50.pt"
    ),
    "final_best_e4b (max_norm_window pooling, 'angry' dominated)": Path(
        "vectors/emotion_vectors_clear_maxnorm_e4b.pt"
    ),
}


def md_norm_table(norms: dict[str, float]) -> str:
    sorted_items = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["| emotion | L2 norm |", "| --- | ---: |"]
    for emotion, norm in sorted_items:
        lines.append(f"| {emotion} | {norm:.3f} |")
    return "\n".join(lines)


def md_cosine_matrix(emotions: list[str], cos: torch.Tensor) -> str:
    header = "| | " + " | ".join(emotions) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(emotions)) + " |"
    rows = [header, sep]
    for i, emotion in enumerate(emotions):
        cells = [f"{cos[i, j].item():.3f}" for j in range(len(emotions))]
        rows.append(f"| **{emotion}** | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def analyze(label: str, path: Path) -> None:
    payload = load_vector_payload(path)
    vectors = payload["vectors"]
    emotions = sorted(vectors)
    matrix = torch.stack([vectors[e].float() for e in emotions])

    norms = {e: float(matrix[i].norm().item()) for i, e in enumerate(emotions)}
    sorted_norms = sorted(norms.values())
    n = len(sorted_norms)
    median = (
        sorted_norms[n // 2]
        if n % 2 == 1
        else 0.5 * (sorted_norms[n // 2 - 1] + sorted_norms[n // 2])
    )
    largest_emotion = max(norms, key=norms.get)
    largest_norm = norms[largest_emotion]
    ratio = largest_norm / median

    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = normalized @ normalized.T

    eye = torch.eye(len(emotions), dtype=torch.bool)
    off = cos[~eye]
    mean_off = float(off.mean().item())
    max_off_idx = int(off.argmax().item())
    flat_idx = (~eye).nonzero(as_tuple=False)[max_off_idx].tolist()
    max_off = float(off.max().item())
    max_off_pair = (emotions[flat_idx[0]], emotions[flat_idx[1]])

    print(f"## {label}")
    print()
    print(f"- source: `{path}`")
    print(
        f"- pooling_strategy: `{payload.get('pooling_strategy')}`, "
        f"layer_idx: `{payload.get('layer_idx')}`, "
        f"start_token: `{payload.get('start_token')}`, "
        f"vector dim: `{matrix.shape[1]}`"
    )
    print(f"- largest-norm emotion: **{largest_emotion}** ({largest_norm:.3f})")
    print(f"- median norm: {median:.3f}")
    print(f"- largest / median ratio: **{ratio:.2f}×**")
    print(f"- mean off-diagonal cosine: **{mean_off:.3f}**")
    print(
        f"- max off-diagonal cosine: **{max_off:.3f}** "
        f"({max_off_pair[0]} ↔ {max_off_pair[1]})"
    )
    print()
    print("### L2 norms per emotion")
    print()
    print(md_norm_table(norms))
    print()
    print("### Pairwise cosine similarity")
    print()
    print(md_cosine_matrix(emotions, cos))
    print()


def main() -> None:
    for label, path in RUNS.items():
        analyze(label, path)


if __name__ == "__main__":
    main()
