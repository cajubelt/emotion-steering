#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.vector_payloads import load_vector_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two vector payload files.")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    return parser.parse_args()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-8))


def main() -> None:
    args = parse_args()
    payload_a = load_vector_payload(args.a)
    payload_b = load_vector_payload(args.b)
    vectors_a = payload_a["vectors"]
    vectors_b = payload_b["vectors"]

    shared_emotions = sorted(set(vectors_a) & set(vectors_b))
    only_in_a = sorted(set(vectors_a) - set(vectors_b))
    only_in_b = sorted(set(vectors_b) - set(vectors_a))

    per_emotion = {}
    cosine_scores = []
    relative_l2_scores = []
    for emotion in shared_emotions:
        vector_a = vectors_a[emotion].float().cpu()
        vector_b = vectors_b[emotion].float().cpu()
        l2_diff = float((vector_a - vector_b).norm().item())
        relative_l2 = float(l2_diff / vector_a.norm().clamp_min(1e-8).item())
        cosine = cosine_similarity(vector_a, vector_b)
        cosine_scores.append(cosine)
        relative_l2_scores.append(relative_l2)
        per_emotion[emotion] = {
            "cosine_similarity": cosine,
            "relative_l2": relative_l2,
            "norm_a": float(vector_a.norm().item()),
            "norm_b": float(vector_b.norm().item()),
            "max_abs_diff": float((vector_a - vector_b).abs().max().item()),
        }

    summary = {
        "a": str(args.a),
        "b": str(args.b),
        "metadata": {
            "layer_idx_a": payload_a["layer_idx"],
            "layer_idx_b": payload_b["layer_idx"],
            "construction_mode_a": payload_a.get("construction_mode", "grand_mean"),
            "construction_mode_b": payload_b.get("construction_mode", "grand_mean"),
            "pooling_strategy_a": payload_a.get("pooling_strategy", "mean"),
            "pooling_strategy_b": payload_b.get("pooling_strategy", "mean"),
        },
        "shared_emotions": shared_emotions,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "aggregate": {
            "mean_cosine_similarity": sum(cosine_scores) / len(cosine_scores) if cosine_scores else None,
            "mean_relative_l2": sum(relative_l2_scores) / len(relative_l2_scores) if relative_l2_scores else None,
        },
        "per_emotion": per_emotion,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
