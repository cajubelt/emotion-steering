#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.io_utils import write_json
from llm_emotions.modeling import apply_residual_steering, generate_text, load_model
from llm_emotions.vector_payloads import load_vector_payload


DEFAULT_PROBE_PROMPTS = {
    "sad": "Write a realistic one-paragraph message from someone leaving an apartment after a difficult ending.",
    "afraid": "Write a realistic one-paragraph message from someone who hears an unexpected noise outside their door late at night.",
    "angry": "Write a realistic one-paragraph message to a manager named Priya after a coworker named Jamie took credit for the sender's Q3 analysis again.",
    "reflective": "Write a realistic one-paragraph journal entry from someone rereading an old letter.",
    "nervous": "Write a realistic one-paragraph message from someone waiting to walk into an important interview.",
    "surprised": "Write a realistic one-paragraph message from someone who just learned unexpected news from an old friend.",
    "calm": "Write a realistic one-paragraph description of someone spending a quiet afternoon at home while rain falls outside.",
    "hopeful": "Write a realistic one-paragraph message from someone who just received the first promising sign after a long setback.",
    "excited": "Write a realistic one-paragraph message from someone who just got two October train tickets to Kyoto after wanting to visit for years.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noninteractive steering probes for an emotion-vector payload.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--emotions")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def parse_emotions(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    emotions = [chunk.strip() for chunk in raw_value.split(",") if chunk.strip()]
    return emotions or None


def generate_probe_text(
    loaded,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful writing assistant. Answer directly with concrete details and no bracketed placeholders.",
        },
        {"role": "user", "content": prompt},
    ]
    return generate_text(
        loaded,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=False,
    ).strip()


def main() -> None:
    args = parse_args()
    payload = load_vector_payload(args.vectors)
    vectors = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    requested_emotions = parse_emotions(args.emotions) or sorted(vectors)
    unknown_emotions = sorted(set(requested_emotions) - set(vectors))
    if unknown_emotions:
        raise ValueError(f"Unknown emotions for vector payload: {unknown_emotions}")
    missing_prompts = sorted(set(requested_emotions) - set(DEFAULT_PROBE_PROMPTS))
    if missing_prompts:
        raise ValueError(f"Missing default probe prompts for emotions: {missing_prompts}")

    loaded = load_model(model_id=args.model_id)
    results = []
    for emotion in requested_emotions:
        print(f"Running steering probe for {emotion}")
        prompt = DEFAULT_PROBE_PROMPTS[emotion]
        base = generate_probe_text(
            loaded,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        with apply_residual_steering(
            loaded,
            layer_idx=layer_idx,
            steering_vectors=[(vectors[emotion], args.strength)],
        ):
            positive = generate_probe_text(
                loaded,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        with apply_residual_steering(
            loaded,
            layer_idx=layer_idx,
            steering_vectors=[(vectors[emotion], -args.strength)],
        ):
            negative = generate_probe_text(
                loaded,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        results.append(
            {
                "emotion": emotion,
                "prompt": prompt,
                "base": base,
                "positive": positive,
                "negative": negative,
            }
        )

    output = {
        "model_id": args.model_id,
        "vector_path": str(args.vectors),
        "layer_idx": layer_idx,
        "strength": args.strength,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(f"Wrote steering probes for {len(results)} emotions to {args.output}")


if __name__ == "__main__":
    main()
