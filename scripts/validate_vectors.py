#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID, REPORTS_DIR
from llm_emotions.io_utils import write_json
from llm_emotions.modeling import capture_hidden_mean, load_model
from llm_emotions.vector_payloads import load_vector_payload


NATURAL_PROMPTS = {
    "happy": "My daughter just took her first steps today. What are some ways to celebrate this with our family?",
    "proud": "My son just graduated top of his class after years of struggling. How should we celebrate?",
    "calm": "I'm sitting with tea and listening to the rain. What's a good book for a quiet afternoon?",
    "desperate": "I've been unemployed for months, my savings are gone, and I just got an eviction notice. What do I do?",
    "angry": "My coworker took credit for my work again and I feel like screaming. How do I respond?",
    "guilty": "I forgot my mother's birthday and she found out I was out partying instead. I feel awful.",
    "sad": "My dog passed away this morning after fourteen years with me. I don't know what to do with his things.",
    "afraid": "There is someone trying to break into my apartment right now and my phone is dying. What should I do?",
    "nervous": "I have a job interview tomorrow for my dream role and I can't stop imagining everything going wrong.",
    "surprised": "My best friend of twenty years just confessed that much of their life story was made up. How do I process that?",
}

MATCHED_PROMPT_FAMILIES = {
    "family_dinner": "At a family dinner, a character feels {emotion}. Describe the next moment.",
    "waiting_room": "In a waiting room, a character feels {emotion}. Describe what they notice and do next.",
    "unexpected_news": "A character receives unexpected news and feels {emotion}. Describe the immediate response.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted emotion vectors.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--scoring",
        choices=["dot", "cosine", "centered_cosine"],
        default="cosine",
        help=(
            "How to score prompt activations against emotion vectors. "
            "'cosine' (default) reproduces the historical reports. "
            "'dot' is an unnormalised dot product. "
            "'centered_cosine' subtracts the mean held-out prompt activation "
            "before cosine scoring (diagnostic for prompt-side common-mode bias)."
        ),
    )
    return parser.parse_args()


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ right_vectors[:2].T
    explained = singular_values.pow(2)
    explained_ratio = explained / explained.sum()
    return coords, explained_ratio[:2]


def compute_projection_scores(
    activation: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    *,
    scoring: str = "cosine",
    activation_offset: torch.Tensor | None = None,
) -> dict[str, float]:
    a = activation.float()
    if scoring == "centered_cosine":
        if activation_offset is None:
            raise ValueError("centered_cosine requires activation_offset")
        a = a - activation_offset.float()

    if scoring == "dot":
        return {
            emotion: float(torch.dot(a, vector.float()).item())
            for emotion, vector in vectors.items()
        }

    a_normalized = a / a.norm().clamp_min(1e-8)
    projection_scores = {}
    for emotion, vector in vectors.items():
        v = vector.float()
        v_normalized = v / v.norm().clamp_min(1e-8)
        projection_scores[emotion] = float(torch.dot(a_normalized, v_normalized).item())
    return projection_scores


def ranked_matches(projection_scores: dict[str, float], top_k: int = 5) -> list[list[object]]:
    ranked = sorted(projection_scores.items(), key=lambda item: item[1], reverse=True)
    return [[emotion, score] for emotion, score in ranked[:top_k]]


def emotion_rank(top_matches: list[list[object]], expected_emotion: str) -> int | None:
    return next((idx + 1 for idx, (emotion, _) in enumerate(top_matches) if emotion == expected_emotion), None)


def summarize_probe_results(
    rows: list[dict],
    *,
    available_emotions: set[str],
) -> dict:
    evaluated_rows = [row for row in rows if row["expected_emotion"] in available_emotions]
    ranks = []
    by_emotion: dict[str, list[int]] = {}

    for row in evaluated_rows:
        rank = emotion_rank(row["top_matches"], row["expected_emotion"])
        if rank is None:
            rank = len(row["top_matches"]) + 1
        ranks.append(rank)
        by_emotion.setdefault(row["expected_emotion"], []).append(rank)

    summary = {
        "evaluated_prompts": len(evaluated_rows),
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "mean_rank": None,
        "mean_reciprocal_rank": None,
        "by_emotion": {},
    }
    if not ranks:
        return summary

    summary["hit_at_1"] = sum(1 for rank in ranks if rank <= 1) / len(ranks)
    summary["hit_at_3"] = sum(1 for rank in ranks if rank <= 3) / len(ranks)
    summary["hit_at_5"] = sum(1 for rank in ranks if rank <= 5) / len(ranks)
    summary["mean_rank"] = sum(ranks) / len(ranks)
    summary["mean_reciprocal_rank"] = sum(1.0 / rank for rank in ranks) / len(ranks)

    for emotion, emotion_ranks in sorted(by_emotion.items()):
        summary["by_emotion"][emotion] = {
            "count": len(emotion_ranks),
            "mean_rank": sum(emotion_ranks) / len(emotion_ranks),
            "hit_at_1": sum(1 for rank in emotion_ranks if rank <= 1) / len(emotion_ranks),
            "hit_at_3": sum(1 for rank in emotion_ranks if rank <= 3) / len(emotion_ranks),
        }

    return summary


def main() -> None:
    args = parse_args()
    payload = load_vector_payload(args.vectors)
    vectors: dict[str, torch.Tensor] = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    start_token = int(payload.get("start_token", 0))
    prompt_probe_start_token = 0
    pooling_strategy = payload.get("pooling_strategy", "mean")
    pool_size = int(payload.get("pool_size", 32))
    available_emotions = set(vectors)

    loaded = load_model(model_id=args.model_id)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    vocab_matrix = loaded.model.lm_head.weight.detach().float().cpu()
    tokenizer = loaded.processor

    logit_lens = {}
    for emotion, vector in vectors.items():
        logits = vocab_matrix @ vector.float()
        top_ids = torch.topk(logits, k=args.top_k).indices.tolist()
        bottom_ids = torch.topk(-logits, k=args.top_k).indices.tolist()
        logit_lens[emotion] = {
            "top_tokens": [tokenizer.decode([token_id]).strip() for token_id in top_ids],
            "bottom_tokens": [tokenizer.decode([token_id]).strip() for token_id in bottom_ids],
        }
    write_json(args.report_dir / "logit_lens.json", logit_lens)

    natural_records = []
    for expected_emotion, prompt in NATURAL_PROMPTS.items():
        activation = capture_hidden_mean(
            loaded,
            prompt,
            layer_idx=layer_idx,
            start_token=prompt_probe_start_token,
            pooling_strategy=pooling_strategy,
            pool_size=pool_size,
        )
        natural_records.append((expected_emotion, prompt, activation))

    matched_records = []
    for family_id, template in MATCHED_PROMPT_FAMILIES.items():
        for expected_emotion in sorted(vectors):
            prompt = template.format(emotion=expected_emotion)
            activation = capture_hidden_mean(
                loaded,
                prompt,
                layer_idx=layer_idx,
                start_token=prompt_probe_start_token,
                pooling_strategy=pooling_strategy,
                pool_size=pool_size,
            )
            matched_records.append((family_id, expected_emotion, prompt, activation))

    activation_offset = None
    if args.scoring == "centered_cosine":
        all_activations = torch.stack(
            [rec[2] for rec in natural_records] + [rec[3] for rec in matched_records]
        )
        activation_offset = all_activations.mean(dim=0)

    prompt_results = []
    for expected_emotion, prompt, activation in natural_records:
        projection_scores = compute_projection_scores(
            activation,
            vectors,
            scoring=args.scoring,
            activation_offset=activation_offset,
        )
        prompt_results.append(
            {
                "expected_emotion": expected_emotion,
                "prompt": prompt,
                "top_matches": ranked_matches(projection_scores),
            }
        )
    write_json(args.report_dir / "prompt_probe_results.json", prompt_results)

    matched_prompt_results = []
    for family_id, expected_emotion, prompt, activation in matched_records:
        projection_scores = compute_projection_scores(
            activation,
            vectors,
            scoring=args.scoring,
            activation_offset=activation_offset,
        )
        matched_prompt_results.append(
            {
                "family_id": family_id,
                "expected_emotion": expected_emotion,
                "prompt": prompt,
                "top_matches": ranked_matches(projection_scores),
            }
        )
    write_json(args.report_dir / "matched_prompt_probe_results.json", matched_prompt_results)

    prompt_probe_summary = {
        "config": {
            "layer_idx": layer_idx,
            "start_token": start_token,
            "prompt_probe_start_token": prompt_probe_start_token,
            "pooling_strategy": pooling_strategy,
            "pool_size": pool_size,
            "construction_mode": payload.get("construction_mode", "grand_mean"),
            "pairwise_preset": payload.get("pairwise_preset"),
            "num_vectors": len(vectors),
            "scoring": args.scoring,
        },
        "natural": summarize_probe_results(prompt_results, available_emotions=available_emotions),
        "matched": summarize_probe_results(matched_prompt_results, available_emotions=available_emotions),
    }
    write_json(args.report_dir / "prompt_probe_summary.json", prompt_probe_summary)

    emotions = sorted(vectors)
    vector_matrix = torch.stack([vectors[emotion].float() for emotion in emotions])
    coords, explained = pca_2d(vector_matrix)

    geometry_rows = []
    for idx, emotion in enumerate(emotions):
        geometry_rows.append(
            {
                "emotion": emotion,
                "pc1": float(coords[idx, 0].item()),
                "pc2": float(coords[idx, 1].item()),
            }
        )
    csv_path = args.report_dir / "emotion_geometry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pc1", "pc2"])
        writer.writeheader()
        writer.writerows(geometry_rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0].numpy(), coords[:, 1].numpy())
    for idx, emotion in enumerate(emotions):
        ax.annotate(emotion, (coords[idx, 0].item(), coords[idx, 1].item()))
    ax.set_title("Emotion vectors projected to first two principal components")
    ax.set_xlabel(f"PC1 ({explained[0].item():.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1].item():.1%} variance)")
    fig.tight_layout()
    fig.savefig(args.report_dir / "emotion_geometry.png", dpi=200)

    print(json.dumps(prompt_probe_summary, indent=2))
    print(f"Wrote validation artifacts to {args.report_dir}")


if __name__ == "__main__":
    main()
