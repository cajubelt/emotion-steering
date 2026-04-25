#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.modeling import (
    capture_hidden_mean,
    generate_text,
    get_decoder_layers,
    load_model,
    locate_decoder_layers_container,
    target_layer_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Gemma once and verify the core steering hooks work.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_model(model_id=args.model_id)

    decoder_path, _ = locate_decoder_layers_container(loaded.model)
    layers = get_decoder_layers(loaded.model)
    layer_idx = target_layer_index(loaded.model)

    print(f"Loaded model: {args.model_id}")
    print(f"Wrapper type: {type(loaded.model).__name__}")
    print(f"Decoder path: {decoder_path}.layers")
    print(f"Decoder layers: {len(layers)}")
    print(f"Target steering layer: {layer_idx}")

    activation = capture_hidden_mean(
        loaded,
        "A person quietly reflects on a difficult week while waiting for the rain to stop.",
        layer_idx=layer_idx,
        start_token=0,
    )
    print(f"Captured activation shape: {tuple(activation.shape)}")
    print(f"Captured activation norm: {activation.norm().item():.4f}")

    if args.skip_generation:
        return

    response = generate_text(
        loaded,
        [
            {
                "role": "system",
                "content": "You are a concise assistant.",
            },
            {
                "role": "user",
                "content": "In one sentence, say hello and mention a clear sky.",
            },
        ],
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ).strip()
    print("Generation preview:")
    print(response)


if __name__ == "__main__":
    main()
