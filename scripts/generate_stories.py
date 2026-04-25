#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import (
    DEFAULT_EMOTION_PATH,
    DEFAULT_MODEL_ID,
    DEFAULT_REVIEW_SAMPLE_FRACTION,
    DEFAULT_STORIES_PER_EMOTION,
    DEFAULT_SYSTEM_PROMPT,
)
from llm_emotions.io_utils import load_json, write_jsonl
from llm_emotions.modeling import generate_text, load_model


TOPICS = [
    "a job interview",
    "a family dinner",
    "a long train ride",
    "moving to a new city",
    "the final day of school",
    "a medical waiting room",
    "a surprise phone call",
    "a wedding speech",
    "a quiet afternoon in the rain",
    "an overdue rent payment",
    "a big presentation at work",
    "finding an old letter",
    "returning to a hometown",
    "getting locked out late at night",
    "a competitive sports match",
    "a tense courtroom hallway",
    "a child's birthday party",
    "an airport delay",
    "a broken friendship",
    "a camping trip",
    "a hospital cafeteria",
    "a graduation ceremony",
    "a missed deadline",
    "a first date",
    "an empty apartment after a breakup",
    "a lottery ticket check",
    "a difficult customer service call",
    "a neighborhood block party",
    "a failed audition",
    "watching the sunrise alone",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate emotion-labeled stories with Gemma 4.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--emotions", type=Path, default=DEFAULT_EMOTION_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/stories/stories.jsonl"))
    parser.add_argument("--review-sample", type=Path, default=Path("data/stories/review_sample.jsonl"))
    parser.add_argument("--stories-per-emotion", type=int, default=DEFAULT_STORIES_PER_EMOTION)
    parser.add_argument("--review-fraction", type=float, default=DEFAULT_REVIEW_SAMPLE_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def build_prompt(emotion: str, topic: str, variant_idx: int) -> list[dict]:
    user_prompt = f"""
Write one short third-person story about a character in {topic}.

Requirements:
- The character should clearly experience the emotion: {emotion}.
- Use 4 to 6 sentences.
- Show the emotion through behavior, thoughts, dialogue, and decisions.
- Avoid directly using the exact word "{emotion}" unless it feels unavoidable.
- Make the topic and details concrete and specific.
- Do not add a title or bullet points.
- Variant seed: {variant_idx}
""".strip()
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    emotions = load_json(args.emotions)
    loaded = load_model(model_id=args.model_id)

    rows = []
    total = len(emotions) * args.stories_per_emotion
    progress = tqdm(total=total, desc="Generating stories")

    for emotion in emotions:
        emotion_topics = TOPICS.copy()
        rng.shuffle(emotion_topics)
        for story_idx in range(args.stories_per_emotion):
            topic = emotion_topics[story_idx % len(emotion_topics)]
            messages = build_prompt(emotion, topic, story_idx)
            story = generate_text(
                loaded,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=False,
            ).strip()
            rows.append(
                {
                    "emotion": emotion,
                    "topic": topic,
                    "story_idx": story_idx,
                    "story": story,
                    "model_id": args.model_id,
                }
            )
            progress.update(1)

    progress.close()
    write_jsonl(args.output, rows)

    sample_size = max(1, math.ceil(len(rows) * args.review_fraction))
    review_rows = rng.sample(rows, sample_size)
    write_jsonl(args.review_sample, review_rows)

    print(f"Wrote {len(rows)} stories to {args.output}")
    print(f"Wrote {len(review_rows)} review stories to {args.review_sample}")


if __name__ == "__main__":
    main()
