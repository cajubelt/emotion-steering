#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID
from llm_emotions.modeling import (
    apply_residual_steering,
    generate_text,
    load_model,
)


HELP_TEXT = """
Commands:
  /list                         Show available emotions
  /steer calm=0.05 sad=-0.02   Set steering map
  /clear                        Remove all steering
  /thinking on|off              Toggle Gemma reasoning mode
  /show                         Show current steering
  /quit                         Exit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive emotion-steered chat.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def parse_steering(text: str) -> dict[str, float]:
    pairs = {}
    payload = text[len("/steer") :].strip()
    for chunk in payload.split():
        emotion, raw_strength = chunk.split("=", maxsplit=1)
        pairs[emotion] = float(raw_strength)
    return pairs


def main() -> None:
    args = parse_args()
    payload = torch.load(args.vectors, map_location="cpu")
    vectors: dict[str, torch.Tensor] = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    loaded = load_model(model_id=args.model_id)

    history: list[dict] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers concise unless asked for depth.",
        }
    ]
    active = {}
    enable_thinking = False

    print(HELP_TEXT.strip())
    print(f"Steering layer: {layer_idx}")

    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/list":
            print(", ".join(sorted(vectors)))
            continue
        if user_text == "/clear":
            active = {}
            print("Cleared steering.")
            continue
        if user_text == "/show":
            print(active or "No active steering.")
            continue
        if user_text.startswith("/thinking "):
            enable_thinking = user_text.split(maxsplit=1)[1].lower() == "on"
            print(f"Thinking mode: {'on' if enable_thinking else 'off'}")
            continue
        if user_text.startswith("/steer "):
            proposed = parse_steering(user_text)
            missing = [emotion for emotion in proposed if emotion not in vectors]
            if missing:
                print(f"Unknown emotions: {', '.join(missing)}")
                continue
            active = proposed
            print(f"Active steering: {active}")
            continue

        history.append({"role": "user", "content": user_text})
        steering_vectors = [(vectors[emotion], strength) for emotion, strength in active.items()]
        with apply_residual_steering(loaded, layer_idx=layer_idx, steering_vectors=steering_vectors):
            response = generate_text(
                loaded,
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=enable_thinking,
            )
        history.append({"role": "assistant", "content": response})
        print(f"\nAssistant> {response}")


if __name__ == "__main__":
    main()
