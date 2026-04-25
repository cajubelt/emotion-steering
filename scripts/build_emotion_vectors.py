#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.activation_cache import load_cache_rows, read_cache_manifest
from llm_emotions.vector_construction import (
    PAIRWISE_PRESETS,
    aggregate_emotion_means,
    build_raw_vectors,
    build_vector_payload,
    choose_components,
    group_story_activations,
    project_out_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build emotion vectors from cached activations.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("vectors/emotion_vectors_from_cache.pt"))
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument(
        "--construction-mode",
        choices=["grand_mean", "one_vs_rest", "pairwise"],
        default="grand_mean",
    )
    parser.add_argument("--pairwise-preset", choices=sorted(PAIRWISE_PRESETS), default="clear_pairs")
    parser.add_argument("--variance-threshold", type=float, default=0.50)
    parser.add_argument("--include-emotions")
    parser.add_argument("--include-topics")
    parser.add_argument("--exclude-topics")
    return parser.parse_args()


def layer_offset(layer_indices: list[int], layer_idx: int) -> int:
    if layer_idx not in layer_indices:
        raise ValueError(f"Layer {layer_idx} is not present in the cache.")
    return layer_indices.index(layer_idx)


def parse_csv_arg(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    values = {chunk.strip() for chunk in raw_value.split(",") if chunk.strip()}
    return values or None


def filter_story_rows(
    story_rows: list[dict],
    *,
    include_emotions: set[str] | None,
    include_topics: set[str] | None,
    exclude_topics: set[str] | None,
) -> list[dict]:
    filtered_rows = []
    for row in story_rows:
        if include_emotions is not None and row["emotion"] not in include_emotions:
            continue
        if include_topics is not None and row["topic"] not in include_topics:
            continue
        if exclude_topics is not None and row["topic"] in exclude_topics:
            continue
        filtered_rows.append(row)
    return filtered_rows


def main() -> None:
    args = parse_args()
    include_emotions = parse_csv_arg(args.include_emotions)
    include_topics = parse_csv_arg(args.include_topics)
    exclude_topics = parse_csv_arg(args.exclude_topics)
    manifest = read_cache_manifest(args.cache_dir / "manifest.json")
    cache_layer_indices = manifest["layer_indices"]
    cache_layer_offset = layer_offset(cache_layer_indices, args.layer_idx)

    story_rows = load_cache_rows(args.cache_dir, "stories")
    neutral_rows = load_cache_rows(args.cache_dir, "neutral")
    story_rows = filter_story_rows(
        story_rows,
        include_emotions=include_emotions,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
    )
    if not story_rows:
        raise ValueError("No story rows remain after applying filters.")

    grouped = group_story_activations(
        [
            {
                "emotion": row["emotion"],
                "activation": row["activations"][cache_layer_offset],
            }
            for row in story_rows
        ]
    )
    emotion_means, emotion_counts = aggregate_emotion_means(grouped)
    raw_vectors, contrast_map = build_raw_vectors(
        emotion_means,
        construction_mode=args.construction_mode,
        pairwise_preset=args.pairwise_preset,
    )

    neutral_matrix = torch.stack([row["activations"][cache_layer_offset].float().cpu() for row in neutral_rows])
    components, num_components = choose_components(neutral_matrix, args.variance_threshold)
    denoised_vectors = {
        emotion: project_out_components(vector.clone(), components)
        for emotion, vector in raw_vectors.items()
    }

    payload = build_vector_payload(
        model_id=manifest["model_id"],
        layer_idx=args.layer_idx,
        start_token=int(manifest["start_token"]),
        pooling_strategy=manifest["pooling_strategy"],
        pool_size=int(manifest["pool_size"]),
        construction_mode=args.construction_mode,
        pairwise_preset=args.pairwise_preset,
        raw_vectors=raw_vectors,
        vectors=denoised_vectors,
        emotion_counts=emotion_counts,
        num_components_projected_out=num_components,
        average_residual_norm=float(neutral_matrix.norm(dim=1).mean().item()),
        contrast_map=contrast_map,
        source_cache_dir=str(args.cache_dir),
    )
    payload["filter_config"] = {
        "include_emotions": sorted(include_emotions) if include_emotions is not None else None,
        "include_topics": sorted(include_topics) if include_topics is not None else None,
        "exclude_topics": sorted(exclude_topics) if exclude_topics is not None else None,
    }
    topic_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in story_rows:
        topic_counts[row["emotion"]][row["topic"]] += 1
    payload["topic_counts"] = {emotion: dict(sorted(counts.items())) for emotion, counts in sorted(topic_counts.items())}
    payload["story_count"] = len(story_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"Wrote {len(denoised_vectors)} emotion vectors to {args.output}")
    print(
        f"Built from cache={args.cache_dir}, layer={args.layer_idx}, "
        f"construction={args.construction_mode}, variance_threshold={args.variance_threshold}"
    )


if __name__ == "__main__":
    main()
