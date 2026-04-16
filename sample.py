from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from tokenrail import BatchExecutor, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenAI batch smoke test for tokenrail.")
    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="Low-cost model used for all requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="OpenAI SDK retry count.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Thread count for BatchExecutor.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Cap output tokens to keep the batch cheap.",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_out",
        help="Directory for jsonl and per-request outputs.",
    )
    return parser.parse_args()


def build_queries() -> dict[str, list[dict[str, str]]]:
    return {
        "q1": [{"role": "user", "content": "Reply with exactly: ok"}],
        "q2": [{"role": "user", "content": "Reply with exactly one word: blue"}],
        "q3": [{"role": "user", "content": "Reply with exactly one digit: 7"}],
    }


def main() -> int:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    result_path = output_dir / "results.jsonl"
    per_request_dir = output_dir / "requests"

    client = RailClient.openai(max_retries=args.max_retries)
    items = batch_items_from_queries(
        build_queries(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )
    monitor = RollingMetricsMonitor()
    executor = BatchExecutor(
        client=client,
        max_workers=args.max_workers,
        sinks=[
            ResultsJsonlSink(result_path),
            PerRequestJsonSink(per_request_dir),
        ],
        monitor=monitor,
    )

    print(f"prepared_items={len(items)} model={args.model} output_dir={output_dir}")
    stats = executor.run(items)

    print()
    print("final_stats:")
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    print()
    print("saved_files:")
    print(f"- {result_path}")
    print(f"- {per_request_dir}")
    print()
    print("rerun_note:")
    print("Run the same command again to confirm completed ids are skipped from the sink state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
