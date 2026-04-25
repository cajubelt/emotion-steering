#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt probe summary files.")
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser.parse_args()


def summarize_file(path: Path) -> dict:
    payload = json.load(path.open("r", encoding="utf-8"))
    return {
        "path": str(path),
        "natural_hit_at_3": payload["natural"]["hit_at_3"],
        "natural_hit_at_5": payload["natural"]["hit_at_5"],
        "natural_mrr": payload["natural"]["mean_reciprocal_rank"],
        "matched_hit_at_3": payload["matched"]["hit_at_3"],
        "matched_hit_at_5": payload["matched"]["hit_at_5"],
        "matched_mrr": payload["matched"]["mean_reciprocal_rank"],
    }


def main() -> None:
    args = parse_args()
    rows = [summarize_file(path) for path in args.summaries]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
