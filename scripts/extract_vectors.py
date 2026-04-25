#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.io_utils import read_jsonl, write_jsonl
from llm_emotions.modeling import capture_hidden_mean, load_model, target_layer_index
from llm_emotions.vector_construction import (
    PAIRWISE_PRESETS,
    aggregate_emotion_means,
    build_neutral_rows,
    build_raw_vectors,
    build_vector_payload,
    choose_components,
    project_out_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean-difference emotion vectors from Gemma activations.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("vectors/emotion_vectors.pt"))
    parser.add_argument("--neutral-output", type=Path, default=Path("data/stories/neutral_stories.jsonl"))
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--start-token", type=int, default=50)
    parser.add_argument(
        "--pooling-strategy",
        choices=["mean", "suffix", "max_norm_window"],
        default="mean",
    )
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument(
        "--construction-mode",
        choices=["grand_mean", "one_vs_rest", "pairwise"],
        default="grand_mean",
    )
    parser.add_argument("--pairwise-preset", choices=sorted(PAIRWISE_PRESETS), default="clear_pairs")
    parser.add_argument("--neutral-count", type=int, default=200)
    parser.add_argument("--variance-threshold", type=float, default=0.50)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    loaded = load_model(model_id=args.model_id)
    layer_idx = args.layer_idx if args.layer_idx is not None else target_layer_index(loaded.model)

    rows = list(read_jsonl(args.stories))
    by_emotion: dict[str, list[torch.Tensor]] = defaultdict(list)

    for row in tqdm(rows, desc="Capturing story activations"):
        hidden_mean = capture_hidden_mean(
            loaded,
            row["story"],
            layer_idx=layer_idx,
            start_token=args.start_token,
            pooling_strategy=args.pooling_strategy,
            pool_size=args.pool_size,
        )
        by_emotion[row["emotion"]].append(hidden_mean)

    emotion_means, emotion_counts = aggregate_emotion_means(by_emotion)
    raw_vectors, contrast_map = build_raw_vectors(
        emotion_means,
        construction_mode=args.construction_mode,
        pairwise_preset=args.pairwise_preset,
    )

    neutral_rows = build_neutral_rows(args.neutral_count)
    write_jsonl(args.neutral_output, neutral_rows)
    neutral_acts = []
    for row in tqdm(neutral_rows, desc="Capturing neutral activations"):
        neutral_acts.append(
            capture_hidden_mean(
                loaded,
                row["story"],
                layer_idx=layer_idx,
                start_token=args.start_token,
                pooling_strategy=args.pooling_strategy,
                pool_size=args.pool_size,
            )
        )

    neutral_matrix = torch.stack(neutral_acts)
    components, num_components = choose_components(neutral_matrix, args.variance_threshold)
    denoised_vectors = {
        emotion: project_out_components(vector.clone(), components)
        for emotion, vector in raw_vectors.items()
    }

    average_residual_norm = float(neutral_matrix.norm(dim=1).mean().item())
    payload = build_vector_payload(
        model_id=args.model_id,
        layer_idx=layer_idx,
        start_token=args.start_token,
        pooling_strategy=args.pooling_strategy,
        pool_size=args.pool_size,
        construction_mode=args.construction_mode,
        pairwise_preset=args.pairwise_preset,
        raw_vectors=raw_vectors,
        vectors=denoised_vectors,
        emotion_counts=emotion_counts,
        num_components_projected_out=num_components,
        average_residual_norm=average_residual_norm,
        contrast_map=contrast_map,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"Wrote {len(denoised_vectors)} emotion vectors to {args.output}")
    print(
        f"Used layer index {layer_idx}, pooling={args.pooling_strategy}, "
        f"pool_size={args.pool_size}, construction={args.construction_mode}, "
        f"and projected out {num_components} neutral PCs"
    )


if __name__ == "__main__":
    main()
