#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.io_utils import load_json


REQUIRED_FIELDS = {
    "emotion",
    "vad_cell",
    "topic",
    "story_idx",
    "story",
    "generator_model",
}
SENTENCE_PATTERN = re.compile(r"[.!?]+(?:[\"')\]]+)?(?:\s+|$)")
ALPHABETIC_PATTERN = re.compile(r"[A-Za-z]")
WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")
MIN_WORDS_FOR_REPETITION_CHECK = 20
MAX_MOST_COMMON_WORD_FRACTION = 0.20
MIN_UNIQUE_WORD_RATIO = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cheap deterministic checks over an emotion-story JSONL corpus.")
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--emotions", type=Path, required=True)
    parser.add_argument("--expected-count-per-emotion", type=int, default=100)
    parser.add_argument("--generator-model", default="gpt-5.4-mini")
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-sentences", type=int, default=4)
    parser.add_argument("--max-sentences", type=int, default=8)
    return parser.parse_args()


def sentence_count(story: str) -> int:
    return len(SENTENCE_PATTERN.findall(story.strip()))


def mostly_repeated(story: str) -> bool:
    words = [match.group(0).lower() for match in WORD_PATTERN.finditer(story)]
    if len(words) < MIN_WORDS_FOR_REPETITION_CHECK:
        return False
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    unique_ratio = len(counts) / len(words)
    return (
        most_common_count / len(words) > MAX_MOST_COMMON_WORD_FRACTION
        or unique_ratio < MIN_UNIQUE_WORD_RATIO
    )


def validate_row(
    row: dict,
    *,
    line_number: int,
    valid_emotions: set[str],
    args: argparse.Namespace,
) -> list[str]:
    errors = []
    missing_fields = sorted(REQUIRED_FIELDS - set(row))
    if missing_fields:
        errors.append(f"line {line_number}: missing fields {missing_fields}")
        return errors

    emotion = row["emotion"]
    story = row["story"]
    story_idx = row["story_idx"]

    if emotion not in valid_emotions:
        errors.append(f"line {line_number}: unknown emotion {emotion!r}")
    if not isinstance(story_idx, int):
        errors.append(f"line {line_number}: story_idx must be an integer")
    if row["generator_model"] != args.generator_model:
        errors.append(f"line {line_number}: generator_model must be {args.generator_model!r}")
    if not isinstance(story, str) or not story.strip():
        errors.append(f"line {line_number}: story must be a nonempty string")
        return errors

    character_count = len(story)
    if character_count < args.min_chars or character_count > args.max_chars:
        errors.append(
            f"line {line_number}: story length {character_count} outside "
            f"{args.min_chars}-{args.max_chars}"
        )

    detected_sentences = sentence_count(story)
    if detected_sentences < args.min_sentences or detected_sentences > args.max_sentences:
        errors.append(
            f"line {line_number}: detected {detected_sentences} sentences outside "
            f"{args.min_sentences}-{args.max_sentences}"
        )

    if ALPHABETIC_PATTERN.search(story) is None:
        errors.append(f"line {line_number}: story has no alphabetic content")
    if mostly_repeated(story):
        errors.append(f"line {line_number}: story appears mostly repeated")

    return errors


def main() -> None:
    args = parse_args()
    emotions = load_json(args.emotions)
    valid_emotions = set(emotions)
    counts: Counter[str] = Counter()
    seen_indices: dict[str, set[int]] = defaultdict(set)
    errors = []

    with args.stories.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"line {line_number}: blank line")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row must be a JSON object")
                continue

            row_errors = validate_row(
                row,
                line_number=line_number,
                valid_emotions=valid_emotions,
                args=args,
            )
            errors.extend(row_errors)
            if row_errors:
                continue

            emotion = row["emotion"]
            story_idx = row["story_idx"]
            if story_idx in seen_indices[emotion]:
                errors.append(f"line {line_number}: duplicate story_idx {story_idx} for {emotion}")
            seen_indices[emotion].add(story_idx)
            counts[emotion] += 1

    for emotion in emotions:
        count = counts[emotion]
        if count != args.expected_count_per_emotion:
            errors.append(
                f"{emotion}: expected {args.expected_count_per_emotion} rows, found {count}"
            )
        expected_indices = set(range(args.expected_count_per_emotion))
        missing_indices = sorted(expected_indices - seen_indices[emotion])
        extra_indices = sorted(seen_indices[emotion] - expected_indices)
        if missing_indices:
            errors.append(f"{emotion}: missing story_idx values {missing_indices[:10]}")
        if extra_indices:
            errors.append(f"{emotion}: unexpected story_idx values {extra_indices[:10]}")

    total_rows = sum(counts.values())
    summary = {
        "stories": str(args.stories),
        "total_valid_rows": total_rows,
        "counts": {emotion: counts[emotion] for emotion in emotions},
    }

    if errors:
        print(json.dumps({"summary": summary, "errors": errors}, indent=2), file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
