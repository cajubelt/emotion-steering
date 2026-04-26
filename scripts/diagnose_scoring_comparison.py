#!/usr/bin/env python3
"""Stage 2 diagnostic: compare held-out top-1 distributions under three scorings.

For each of the two diagnostic runs we re-extract activations for the exact
held-out prompts that were saved with that run (read from
prompt_probe_results.json and matched_prompt_probe_results.json), then score
each prompt under dot, cosine, and centered_cosine. We report how many prompts
each emotion claimed top-1 under each scoring.

This script intentionally re-uses the existing held-out prompt sets — it does
not regenerate stories or re-extract vectors.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.modeling import capture_hidden_mean, load_model
from llm_emotions.vector_payloads import load_vector_payload
from validate_vectors import compute_projection_scores


RUNS = [
    {
        "label": "vad_grid_8_e4b_mean50",
        "headline": "mean pooling, 'nervous' dominated",
        "vectors_path": Path("vectors/vad_grid_8_e4b_mean50.pt"),
        "report_dir": Path("reports/vad_grid_8_e4b_mean50"),
    },
    {
        "label": "final_best_e4b",
        "headline": "max_norm_window pooling, 'angry' dominated",
        "vectors_path": Path("vectors/emotion_vectors_clear_maxnorm_e4b.pt"),
        "report_dir": Path("reports/final_best_e4b"),
    },
]


def load_held_out_prompts(report_dir: Path) -> list[dict]:
    prompts: list[dict] = []
    natural_path = report_dir / "prompt_probe_results.json"
    matched_path = report_dir / "matched_prompt_probe_results.json"
    for row in json.loads(natural_path.read_text()):
        prompts.append(
            {
                "kind": "natural",
                "expected_emotion": row["expected_emotion"],
                "prompt": row["prompt"],
            }
        )
    for row in json.loads(matched_path.read_text()):
        prompts.append(
            {
                "kind": "matched",
                "expected_emotion": row["expected_emotion"],
                "family_id": row.get("family_id"),
                "prompt": row["prompt"],
            }
        )
    return prompts


def score_prompts(
    activations: list[torch.Tensor],
    vectors: dict[str, torch.Tensor],
    scoring: str,
) -> list[str]:
    activation_offset = None
    if scoring == "centered_cosine":
        activation_offset = torch.stack(activations).mean(dim=0)
    winners = []
    for activation in activations:
        scores = compute_projection_scores(
            activation,
            vectors,
            scoring=scoring,
            activation_offset=activation_offset,
        )
        winner = max(scores.items(), key=lambda kv: kv[1])[0]
        winners.append(winner)
    return winners


def md_table(emotions: list[str], counts_by_scoring: dict[str, Counter], total: int) -> str:
    scorings = ["dot", "cosine", "centered_cosine"]
    header = "| emotion (top-1 winner) | " + " | ".join(scorings) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(scorings)) + " |"
    rows = [header, sep]
    for emotion in emotions:
        cells = []
        for scoring in scorings:
            c = counts_by_scoring[scoring].get(emotion, 0)
            cells.append(f"{c} ({c / total:.0%})" if c else "0")
        rows.append(f"| {emotion} | " + " | ".join(cells) + " |")
    rows.append(f"| **total prompts** | **{total}** | **{total}** | **{total}** |")
    return "\n".join(rows)


def analyze(loaded, run: dict) -> None:
    payload = load_vector_payload(run["vectors_path"])
    vectors: dict[str, torch.Tensor] = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    pooling_strategy = payload.get("pooling_strategy", "mean")
    pool_size = int(payload.get("pool_size", 32))
    prompt_probe_start_token = 0

    prompts = load_held_out_prompts(run["report_dir"])
    print(
        f"\n## {run['label']} ({run['headline']})",
        flush=True,
    )
    print(
        f"\n- vectors: `{run['vectors_path']}`"
        f"\n- pooling: `{pooling_strategy}`, layer `{layer_idx}`, pool_size `{pool_size}`"
        f"\n- held-out prompts: {len(prompts)} "
        f"({sum(1 for p in prompts if p['kind'] == 'natural')} natural + "
        f"{sum(1 for p in prompts if p['kind'] == 'matched')} matched)",
        flush=True,
    )

    activations: list[torch.Tensor] = []
    for idx, item in enumerate(prompts):
        activation = capture_hidden_mean(
            loaded,
            item["prompt"],
            layer_idx=layer_idx,
            start_token=prompt_probe_start_token,
            pooling_strategy=pooling_strategy,
            pool_size=pool_size,
        )
        activations.append(activation)
        if (idx + 1) % 10 == 0 or idx + 1 == len(prompts):
            print(f"  ...activations {idx + 1}/{len(prompts)}", flush=True)

    counts_by_scoring: dict[str, Counter] = {}
    for scoring in ["dot", "cosine", "centered_cosine"]:
        winners = score_prompts(activations, vectors, scoring)
        counts_by_scoring[scoring] = Counter(winners)

    emotions = sorted(vectors)
    print("\n### Top-1 winner distribution across all held-out prompts\n")
    print(md_table(emotions, counts_by_scoring, total=len(prompts)))


def main() -> None:
    print(f"Loading model {DEFAULT_MODEL_ID}...", flush=True)
    loaded = load_model(model_id=DEFAULT_MODEL_ID)
    print(f"Model on device: {loaded.device}", flush=True)
    for run in RUNS:
        analyze(loaded, run)


if __name__ == "__main__":
    main()
