#!/usr/bin/env python3
"""Qualitative steering demo: generate one neutral prompt at several strengths.

Loads a vector payload, picks one emotion, and generates a baseline plus
positive- and negative-steered completions at a sweep of strengths so the
emotional inflection can be compared side by side. The seed is fixed so the
unsteered token distribution is the same starting point for every strength.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.modeling import apply_residual_steering, generate_text, load_model
from llm_emotions.vector_payloads import load_vector_payload


DEFAULT_PROMPT = (
    "Write a two-sentence reply to a friend who just texted asking how "
    "your day was. Just the message, nothing else."
)
DEFAULT_STRENGTHS = "-0.15,-0.10,-0.05,0,0.05,0.10,0.15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--strengths", default=DEFAULT_STRENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Don't fix the seed; each generation samples freshly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_vector_payload(args.vectors)
    vectors: dict[str, torch.Tensor] = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    if args.emotion not in vectors:
        raise SystemExit(
            f"Emotion '{args.emotion}' not in vectors. Available: {sorted(vectors)}"
        )

    loaded = load_model(model_id=args.model_id)

    strengths = [float(x) for x in args.strengths.split(",")]
    messages = [
        {"role": "system", "content": "You are a helpful writing assistant."},
        {"role": "user", "content": args.prompt},
    ]

    print(f"# Steering demo: {args.emotion}\n")
    print(f"- vectors: `{args.vectors}` (layer {layer_idx}, emotions: {len(vectors)})")
    print(f"- prompt: {args.prompt!r}")
    print(
        f"- decoding: temperature={args.temperature}, top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}, seed={args.seed}"
    )
    print()

    for strength in strengths:
        if not args.no_seed:
            torch.manual_seed(args.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        steering_vectors = (
            [(vectors[args.emotion], strength)] if strength != 0 else []
        )
        with apply_residual_steering(
            loaded, layer_idx=layer_idx, steering_vectors=steering_vectors
        ):
            response = generate_text(
                loaded,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        label = "baseline" if strength == 0 else f"{args.emotion} = {strength:+.2f}"
        print(f"## {label}\n")
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
