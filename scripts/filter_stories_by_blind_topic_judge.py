#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.io_utils import append_jsonl, load_json, write_jsonl


OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BATCH_SIZE = 25
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_SHUFFLE_SEED = 17
ITEM_ID_WIDTH = 6
BLIND_AUDIT_SUFFIX = "_blind_topic_audit.jsonl"
TOPIC_JUDGMENTS_SUFFIX = "_topic_judgments.jsonl"
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
REQUIRED_FIELDS = {"emotion", "topic"}
SYSTEM_PROMPT = (
    "You are a blind annotation judge for an emotion-story dataset. "
    "Your task is to choose which single emotion a topic most naturally evokes. "
    "Use only the topic text and the provided emotion list."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter emotion-story rows by a blind AI judgment of the topic field."
    )
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--emotions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--judgments-output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=OPENAI_API_BASE)
    parser.add_argument("--api-key-env", default=OPENAI_API_KEY_ENV)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRY_COUNT)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_sibling_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


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


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be a JSON object")
            missing_fields = sorted(REQUIRED_FIELDS - set(row))
            if missing_fields:
                raise ValueError(f"line {line_number}: missing fields {missing_fields}")
            if not isinstance(row["topic"], str) or not row["topic"].strip():
                raise ValueError(f"line {line_number}: topic must be a nonempty string")
            if not isinstance(row["emotion"], str) or not row["emotion"].strip():
                raise ValueError(f"line {line_number}: emotion must be a nonempty string")
            rows.append(row)
    return rows


def load_cached_judgments(path: Path, valid_emotions: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}

    cached = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            topic = row.get("topic")
            selected_emotion = row.get("selected_emotion")
            if not isinstance(topic, str) or not isinstance(selected_emotion, str):
                raise ValueError(f"{path}:{line_number}: cached judgment is missing topic or selected_emotion")
            if selected_emotion not in valid_emotions:
                raise ValueError(f"{path}:{line_number}: cached judgment has invalid emotion {selected_emotion!r}")
            cached[topic] = row
    return cached


def build_response_schema(emotions: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["item_id", "selected_emotion", "confidence", "rationale"],
                    "properties": {
                        "item_id": {"type": "string"},
                        "selected_emotion": {"type": "string", "enum": emotions},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def build_user_prompt(emotions: list[str], batch: list[dict]) -> str:
    topic_lines = "\n".join(f"{item['item_id']}: {item['topic']}" for item in batch)
    emotion_list = ", ".join(emotions)
    return f"""
Emotion choices: {emotion_list}

For each topic below, choose the one emotion from the list that resonates most naturally with the topic.
Judge only the topic wording. Do not infer anything from the item id or order.
If several emotions could fit, choose the closest single match.
Return one label for every item id.

Topics:
{topic_lines}
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
    api_base: str,
    api_key: str,
    model: str,
    emotions: list[str],
    batch: list[dict],
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: int,
) -> dict:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(emotions, batch)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "blind_topic_emotion_labels",
                "strict": True,
                "schema": build_response_schema(emotions),
            }
        },
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def request_batch_with_retries(
    *,
    args: argparse.Namespace,
    api_key: str,
    emotions: list[str],
    batch: list[dict],
) -> dict:
    for attempt_idx in range(args.retries + 1):
        try:
            return call_openai_responses_api(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                emotions=emotions,
                batch=batch,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == HTTP_TOO_MANY_REQUESTS or error.code >= HTTP_SERVER_ERROR_MIN
            if not retryable or attempt_idx == args.retries:
                raise RuntimeError(f"OpenAI API error {error.code}: {body}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt_idx == args.retries:
                raise RuntimeError(f"OpenAI API request failed: {error}") from error

        time.sleep(args.retry_backoff_seconds * (attempt_idx + 1))

    raise RuntimeError("OpenAI API request failed without an exception")


def parse_batch_labels(response: dict, batch: list[dict], valid_emotions: set[str]) -> dict[str, dict]:
    payload = json.loads(extract_output_text(response))
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("response JSON must contain a labels array")

    expected_ids = {item["item_id"] for item in batch}
    parsed = {}
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("label entries must be objects")
        item_id = label.get("item_id")
        selected_emotion = label.get("selected_emotion")
        confidence = label.get("confidence")
        rationale = label.get("rationale")
        if item_id not in expected_ids:
            raise ValueError(f"unexpected item id {item_id!r}")
        if item_id in parsed:
            raise ValueError(f"duplicate item id {item_id!r}")
        if selected_emotion not in valid_emotions:
            raise ValueError(f"invalid selected emotion {selected_emotion!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"{item_id}: confidence must be numeric")
        if confidence < MIN_CONFIDENCE or confidence > MAX_CONFIDENCE:
            raise ValueError(f"{item_id}: confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}")
        if not isinstance(rationale, str):
            raise ValueError(f"{item_id}: rationale must be a string")
        parsed[item_id] = {
            "selected_emotion": selected_emotion,
            "confidence": confidence,
            "rationale": rationale,
        }

    missing_ids = sorted(expected_ids - set(parsed))
    if missing_ids:
        raise ValueError(f"missing item ids: {missing_ids}")
    return parsed


def label_topics(
    *,
    args: argparse.Namespace,
    api_key: str | None,
    emotions: list[str],
    topics: list[str],
    judgments_output: Path,
) -> dict[str, dict]:
    valid_emotions = set(emotions)
    judgments = load_cached_judgments(judgments_output, valid_emotions)
    topics_to_label = [topic for topic in topics if topic not in judgments]
    if topics_to_label and not api_key:
        raise SystemExit(f"Set {args.api_key_env} in the shell or in {args.env_file} before running the blind topic judge.")

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(topics_to_label)

    for batch_start in range(0, len(topics_to_label), args.batch_size):
        batch_topics = topics_to_label[batch_start : batch_start + args.batch_size]
        batch = [
            {"item_id": f"topic_{batch_start + item_idx:0{ITEM_ID_WIDTH}d}", "topic": topic}
            for item_idx, topic in enumerate(batch_topics)
        ]
        response = request_batch_with_retries(
            args=args,
            api_key=api_key,
            emotions=emotions,
            batch=batch,
        )
        labels_by_id = parse_batch_labels(response, batch, valid_emotions)
        for item in batch:
            label = labels_by_id[item["item_id"]]
            row = {
                "topic": item["topic"],
                "selected_emotion": label["selected_emotion"],
                "confidence": label["confidence"],
                "rationale": label["rationale"],
                "judge_model": args.model,
            }
            append_jsonl(judgments_output, row)
            judgments[item["topic"]] = row
        print(f"Labeled {min(batch_start + args.batch_size, len(topics_to_label))}/{len(topics_to_label)} new topics")

    return judgments


def build_outputs(rows: list[dict], judgments: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    filtered_rows = []
    audit_rows = []
    target_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()

    for row in rows:
        judgment = judgments[row["topic"]]
        target_emotion = row["emotion"]
        selected_emotion = judgment["selected_emotion"]
        keep = selected_emotion == target_emotion
        target_counts[target_emotion] += 1
        selected_counts[selected_emotion] += 1
        if keep:
            kept_counts[target_emotion] += 1
            filtered_rows.append(row)
        audit_rows.append(
            {
                "emotion": target_emotion,
                "topic": row["topic"],
                "story_idx": row.get("story_idx"),
                "selected_emotion": selected_emotion,
                "confidence": judgment["confidence"],
                "keep": keep,
                "rationale": judgment["rationale"],
                "judge_model": judgment["judge_model"],
            }
        )

    summary = {
        "input_rows": len(rows),
        "kept_rows": len(filtered_rows),
        "removed_rows": len(rows) - len(filtered_rows),
        "target_counts": {
            emotion: target_counts[emotion] for emotion in sorted(target_counts)
        },
        "kept_counts": {
            emotion: kept_counts[emotion] for emotion in sorted(target_counts)
        },
        "selected_counts": {
            emotion: selected_counts[emotion] for emotion in sorted(target_counts)
        },
    }
    return filtered_rows, audit_rows, summary


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    emotions = load_json(args.emotions)
    if not isinstance(emotions, list) or not all(isinstance(emotion, str) for emotion in emotions):
        raise SystemExit("--emotions must point to a JSON array of emotion strings")

    rows = load_rows(args.stories)
    valid_emotions = set(emotions)
    invalid_emotions = sorted({row["emotion"] for row in rows} - valid_emotions)
    if invalid_emotions:
        raise SystemExit(f"stories include emotions not listed in --emotions: {invalid_emotions}")

    topics = sorted({row["topic"] for row in rows})
    audit_output = args.audit_output or default_sibling_path(args.output, BLIND_AUDIT_SUFFIX)
    judgments_output = args.judgments_output or default_sibling_path(args.output, TOPIC_JUDGMENTS_SUFFIX)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stories": str(args.stories),
                    "emotions": emotions,
                    "rows": len(rows),
                    "unique_topics": len(topics),
                    "output": str(args.output),
                    "audit_output": str(audit_output),
                    "judgments_output": str(judgments_output),
                    "model": args.model,
                    "env_file": str(args.env_file),
                    "api_key_available": bool(resolve_api_key(args)),
                },
                indent=2,
            )
        )
        return

    judgments = label_topics(
        args=args,
        api_key=resolve_api_key(args),
        emotions=emotions,
        topics=topics,
        judgments_output=judgments_output,
    )
    filtered_rows, audit_rows, summary = build_outputs(rows, judgments)
    write_jsonl(args.output, filtered_rows)
    write_jsonl(audit_output, audit_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote filtered stories to {args.output}")
    print(f"Wrote row-level audit to {audit_output}")
    print(f"Wrote topic-level judgments to {judgments_output}")


if __name__ == "__main__":
    main()
