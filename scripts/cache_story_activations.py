#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.activation_cache import (
    ACTIVATION_CACHE_VERSION,
    cache_row_filename,
    parse_layer_indices,
    write_cache_manifest,
    write_cache_row,
)
from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.io_utils import read_jsonl
from llm_emotions.modeling import capture_hidden_means, load_model
from llm_emotions.vector_construction import build_neutral_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache pooled activations for multiple layers in one pass.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-indices", default="18,21,23,25,28")
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument(
        "--pooling-strategy",
        choices=["mean", "suffix", "max_norm_window"],
        default="max_norm_window",
    )
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--neutral-count", type=int, default=200)
    return parser.parse_args()


def write_split_rows(
    loaded,
    rows: list[dict],
    *,
    split: str,
    output_dir: Path,
    layer_indices: list[int],
    start_token: int,
    pooling_strategy: str,
    pool_size: int,
) -> int:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(tqdm(rows, desc=f"Caching {split} activations"), start=1):
        activations_by_layer = capture_hidden_means(
            loaded,
            row["story"],
            layer_indices=layer_indices,
            start_token=start_token,
            pooling_strategy=pooling_strategy,
            pool_size=pool_size,
        )
        activation_tensor = torch.stack([activations_by_layer[layer_idx] for layer_idx in layer_indices])
        payload = {
            "schema_version": ACTIVATION_CACHE_VERSION,
            "split": split,
            "model_id": loaded.model_id,
            "layer_indices": layer_indices,
            "start_token": start_token,
            "pooling_strategy": pooling_strategy,
            "pool_size": pool_size,
            "emotion": row.get("emotion"),
            "topic": row.get("topic"),
            "story_idx": row.get("story_idx"),
            "story": row["story"],
            "activations": activation_tensor,
        }
        write_cache_row(split_dir / cache_row_filename(index), payload)

    return len(rows)


def main() -> None:
    args = parse_args()
    layer_indices = parse_layer_indices(args.layer_indices)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    story_rows = list(read_jsonl(args.stories))
    neutral_rows = build_neutral_rows(args.neutral_count)
    loaded = load_model(model_id=args.model_id)

    story_count = write_split_rows(
        loaded,
        story_rows,
        split="stories",
        output_dir=output_dir,
        layer_indices=layer_indices,
        start_token=args.start_token,
        pooling_strategy=args.pooling_strategy,
        pool_size=args.pool_size,
    )
    neutral_count = write_split_rows(
        loaded,
        neutral_rows,
        split="neutral",
        output_dir=output_dir,
        layer_indices=layer_indices,
        start_token=args.start_token,
        pooling_strategy=args.pooling_strategy,
        pool_size=args.pool_size,
    )

    manifest = {
        "schema_version": ACTIVATION_CACHE_VERSION,
        "model_id": args.model_id,
        "stories_path": str(args.stories),
        "layer_indices": layer_indices,
        "start_token": args.start_token,
        "pooling_strategy": args.pooling_strategy,
        "pool_size": args.pool_size,
        "story_count": story_count,
        "neutral_count": neutral_count,
    }
    write_cache_manifest(output_dir / "manifest.json", manifest)

    print(f"Cached {story_count} story activations and {neutral_count} neutral activations to {output_dir}")


if __name__ == "__main__":
    main()
