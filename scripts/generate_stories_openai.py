#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.io_utils import append_jsonl, load_json, read_jsonl


OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_STORIES_PER_EMOTION = 100
DEFAULT_BATCH_SIZE = 10
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_OUTPUT_TOKENS = 12000
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500
MIN_TOPIC_WORDS = 3
MAX_TOPIC_WORDS = 12
MIN_STORY_CHARS = 250
MAX_STORY_CHARS = 1200
MIN_SENTENCES = 4
MAX_SENTENCES = 8
SENTENCE_PATTERN = re.compile(r"[.!?]+(?:[\"')\]]+)?(?:\s+|$)")
WORD_PATTERN = re.compile(r"[A-Za-z]+")
REPETITION_WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")
MIN_WORDS_FOR_REPETITION_CHECK = 20
MAX_MOST_COMMON_WORD_FRACTION = 0.20
MIN_UNIQUE_WORD_RATIO = 0.35
SYSTEM_PROMPT = (
    "You write concrete short fiction examples for an emotion-representation dataset. "
    "Each example must be natural, specific, and varied."
)
VAD_CELLS = {
    "sad": "low_valence_low_arousal",
    "afraid": "low_valence_high_arousal",
    "angry": "low_valence_high_arousal",
    "reflective": "mid_valence_low_arousal",
    "nervous": "mid_valence_high_arousal",
    "calm": "high_valence_low_arousal",
    "hopeful": "high_valence_mid_arousal",
    "excited": "high_valence_high_arousal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate emotion-story corpora with an OpenAI model.")
    parser.add_argument("--emotions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=OPENAI_API_BASE)
    parser.add_argument("--api-key-env", default=OPENAI_API_KEY_ENV)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--stories-per-emotion", type=int, default=DEFAULT_STORIES_PER_EMOTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRY_COUNT)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"{path}:{line_number}: environment key is empty")
            values[key] = parse_env_value(value)
    return values


def resolve_api_key(args: argparse.Namespace) -> str | None:
    return os.environ.get(args.api_key_env) or load_env_file(args.env_file).get(args.api_key_env)


def build_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["stories"],
        "properties": {
            "stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["story_idx", "topic", "story"],
                    "properties": {
                        "story_idx": {"type": "integer"},
                        "topic": {"type": "string"},
                        "story": {"type": "string"},
                    },
                },
            }
        },
    }


def build_user_prompt(emotion: str, emotions: list[str], story_indices: list[int]) -> str:
    choices = ", ".join(emotions)
    indices = ", ".join(str(index) for index in story_indices)
    return f"""
Emotion choices in the experiment: {choices}
Target emotion for this batch: {emotion}
Required story_idx values: {indices}

For each required story_idx, create one dataset row with:
- a concise topic phrase, 3 to 12 words, that would naturally cue the target emotion to a blind judge choosing from the emotion choices;
- a concrete third-person story of 4 to 6 sentences;
- varied characters, settings, sentence shapes, and outcomes across rows.

Avoid using the target emotion word in the topic phrase.
The story may use the target emotion word naturally, but it should also show the feeling through behavior, thoughts, dialogue, and decisions.
Keep pronouns consistent within each story.
Do not write titles, bullets, placeholders, or repeated templates.
Return exactly one story for every required story_idx.
""".strip()


def extract_output_text(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    fragments = []
    for output_item in response.get("output", []):
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if isinstance(text, str):
                fragments.append(text)
    if fragments:
        return "".join(fragments)

    raise ValueError("response did not include output text")


def call_openai_responses_api(
    *,
    args: argparse.Namespace,
    api_key: str,
    emotion: str,
    emotions: list[str],
    story_indices: list[int],
) -> dict:
    payload = {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(emotion, emotions, story_indices)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "emotion_story_batch",
                "strict": True,
                "schema": build_response_schema(),
            }
        },
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
    }
    request = urllib.request.Request(
        f"{args.api_base.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def request_batch_with_retries(
    *,
    args: argparse.Namespace,
    api_key: str,
    emotion: str,
    emotions: list[str],
    story_indices: list[int],
) -> dict:
    for attempt_idx in range(args.retries + 1):
        try:
            return call_openai_responses_api(
                args=args,
                api_key=api_key,
                emotion=emotion,
                emotions=emotions,
                story_indices=story_indices,
            )
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == HTTP_TOO_MANY_REQUESTS or error.code >= HTTP_SERVER_ERROR_MIN
            if not retryable or attempt_idx == args.retries:
                raise RuntimeError(f"OpenAI API error {error.code}: {body}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt_idx == args.retries:
                raise RuntimeError(f"OpenAI API request failed: {error}") from error

        time.sleep(args.retry_backoff_seconds * (attempt_idx + 1))

    raise RuntimeError("OpenAI API request failed without an exception")


def generate_valid_story_batch(
    *,
    args: argparse.Namespace,
    api_key: str,
    emotion: str,
    emotions: list[str],
    story_indices: list[int],
) -> list[dict]:
    for attempt_idx in range(args.retries + 1):
        response = request_batch_with_retries(
            args=args,
            api_key=api_key,
            emotion=emotion,
            emotions=emotions,
            story_indices=story_indices,
        )
        try:
            return parse_story_batch(response, story_indices)
        except ValueError as error:
            if attempt_idx == args.retries:
                raise
            print(f"Retrying {emotion} {story_indices} after invalid content: {error}")
            time.sleep(args.retry_backoff_seconds * (attempt_idx + 1))

    raise RuntimeError("Story batch validation failed without an exception")


def validate_generated_story(row: dict, expected_indices: set[int]) -> None:
    story_idx = row.get("story_idx")
    topic = row.get("topic")
    story = row.get("story")
    if story_idx not in expected_indices:
        raise ValueError(f"unexpected story_idx {story_idx!r}")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"story_idx {story_idx}: topic must be a nonempty string")
    if not isinstance(story, str) or not story.strip():
        raise ValueError(f"story_idx {story_idx}: story must be a nonempty string")

    topic_word_count = len(WORD_PATTERN.findall(topic))
    if topic_word_count < MIN_TOPIC_WORDS or topic_word_count > MAX_TOPIC_WORDS:
        raise ValueError(f"story_idx {story_idx}: topic has {topic_word_count} words")

    story_length = len(story)
    if story_length < MIN_STORY_CHARS or story_length > MAX_STORY_CHARS:
        raise ValueError(f"story_idx {story_idx}: story has {story_length} characters")

    sentence_count = len(SENTENCE_PATTERN.findall(story.strip()))
    if sentence_count < MIN_SENTENCES or sentence_count > MAX_SENTENCES:
        raise ValueError(f"story_idx {story_idx}: story has {sentence_count} detected sentences")

    words = [match.group(0).lower() for match in REPETITION_WORD_PATTERN.finditer(story)]
    if len(words) >= MIN_WORDS_FOR_REPETITION_CHECK:
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        unique_ratio = len(word_counts) / len(words)
        if (
            most_common_count / len(words) > MAX_MOST_COMMON_WORD_FRACTION
            or unique_ratio < MIN_UNIQUE_WORD_RATIO
        ):
            raise ValueError(f"story_idx {story_idx}: story appears mostly repeated")


def parse_story_batch(response: dict, story_indices: list[int]) -> list[dict]:
    payload = json.loads(extract_output_text(response))
    stories = payload.get("stories")
    if not isinstance(stories, list):
        raise ValueError("response JSON must contain a stories array")
    if len(stories) != len(story_indices):
        raise ValueError(f"expected {len(story_indices)} stories, got {len(stories)}")

    expected_indices = set(story_indices)
    parsed = []
    seen_indices = set()
    for row in stories:
        if not isinstance(row, dict):
            raise ValueError("story entries must be objects")
        validate_generated_story(row, expected_indices)
        story_idx = row["story_idx"]
        if story_idx in seen_indices:
            raise ValueError(f"duplicate story_idx {story_idx}")
        seen_indices.add(story_idx)
        parsed.append(row)

    missing_indices = sorted(expected_indices - seen_indices)
    if missing_indices:
        raise ValueError(f"missing story_idx values {missing_indices}")
    return sorted(parsed, key=lambda row: row["story_idx"])


def load_existing_indices(path: Path, emotions: set[str]) -> dict[str, set[int]]:
    existing = {emotion: set() for emotion in emotions}
    if not path.exists():
        return existing

    for row in read_jsonl(path):
        emotion = row.get("emotion")
        story_idx = row.get("story_idx")
        if emotion in existing and isinstance(story_idx, int):
            existing[emotion].add(story_idx)
    return existing


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.stories_per_emotion < 1:
        raise SystemExit("--stories-per-emotion must be at least 1")

    emotions = load_json(args.emotions)
    if not isinstance(emotions, list) or not all(isinstance(emotion, str) for emotion in emotions):
        raise SystemExit("--emotions must point to a JSON array of emotion strings")

    missing_vad_cells = sorted(set(emotions) - set(VAD_CELLS))
    if missing_vad_cells:
        raise SystemExit(f"missing VAD cell mapping for emotions: {missing_vad_cells}")

    api_key = resolve_api_key(args)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} in the shell or in {args.env_file} before generating stories.")

    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing_indices = load_existing_indices(args.output, set(emotions))
    total_written = 0
    for emotion in emotions:
        missing_indices = [
            story_idx
            for story_idx in range(args.stories_per_emotion)
            if story_idx not in existing_indices[emotion]
        ]
        for batch_start in range(0, len(missing_indices), args.batch_size):
            story_indices = missing_indices[batch_start : batch_start + args.batch_size]
            stories = generate_valid_story_batch(
                args=args,
                api_key=api_key,
                emotion=emotion,
                emotions=emotions,
                story_indices=story_indices,
            )
            for story in stories:
                row = {
                    "emotion": emotion,
                    "vad_cell": VAD_CELLS[emotion],
                    "topic": story["topic"].strip(),
                    "story_idx": story["story_idx"],
                    "story": story["story"].strip(),
                    "generator_model": args.model,
                }
                append_jsonl(args.output, row)
                total_written += 1
            completed_count = args.stories_per_emotion - len(missing_indices) + batch_start + len(story_indices)
            print(f"{emotion}: {completed_count}/{args.stories_per_emotion}")

    print(f"Wrote {total_written} new stories to {args.output}")


if __name__ == "__main__":
    main()
