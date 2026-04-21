from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenrail import BatchExecutor, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local vLLM batch smoke test for tokenrail.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Local vLLM model id used for all requests.",
    )
    parser.add_argument(
        "--family",
        choices=["gemma", "qwen"],
        default="qwen",
        help="Prompt/sampling strategy family.",
    )
    parser.add_argument(
        "--batch-flush-size",
        type=int,
        default=256,
        help="Max prompts buffered into one vLLM generate call for a sampling group.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=64,
        help="Cap output tokens for each request.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable family-specific thinking mode.",
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

    output_dir = Path(args.output_dir)
    result_path = output_dir / "results.jsonl"
    per_request_dir = output_dir / "requests"

    client = RailClient.vllm(
        model_id=args.model,
        family=args.family,
        batch_flush_size=args.batch_flush_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    items = batch_items_from_queries(
        build_queries(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=args.enable_thinking,
    )
    monitor = RollingMetricsMonitor()
    executor = BatchExecutor(
        client=client,
        max_workers=1,
        sinks=[
            ResultsJsonlSink(result_path),
            PerRequestJsonSink(per_request_dir),
        ],
        monitor=monitor,
    )

    print(
        "prepared_items="
        f"{len(items)} model={args.model} family={args.family} "
        f"thinking={args.enable_thinking} output_dir={output_dir}"
    )
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
