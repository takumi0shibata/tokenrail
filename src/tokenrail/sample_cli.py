from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .client import RailClient
from .executor import batch_items_from_queries
from .executor import BatchExecutor
from .monitor import RollingMetricsMonitor
from .sinks import PerRequestJsonSink, ResultsJsonlSink


def _default_provider() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "vllm-server"
    return "vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local vLLM batch smoke test for tokenrail.")
    parser.add_argument(
        "--provider",
        choices=["vllm", "vllm-server"],
        default=_default_provider(),
        help="Use in-process vLLM or an OpenAI-compatible vLLM server. Defaults to vllm-server on Apple Silicon.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM server base URL for --provider vllm-server.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key sent to the OpenAI-compatible vLLM server.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Local vLLM model id used for all requests.",
    )
    parser.add_argument(
        "--family",
        choices=["gemma", "qwen"],
        default="qwen",
        help="Prompt/sampling strategy family for in-process vLLM.",
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for each request.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length for in-process vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of CUDA GPU memory reserved by in-process vLLM.",
    )
    parser.add_argument(
        "--metal-memory-fraction",
        default=None,
        help="Apple Silicon vllm-metal memory fraction for in-process vLLM.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional in-process vLLM device argument, for example cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM dtype argument.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional max_num_seqs engine argument for in-process vLLM.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable family-specific thinking mode when supported.",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_out",
        help="Directory for jsonl and per-request outputs.",
    )
    return parser.parse_args()


def build_queries() -> dict[str, list[dict[str, str]]]:
    return {
        "q1": [{"role": "user", "content": "Output only this lowercase token: ok"}],
        "q2": [{"role": "user", "content": "Output only this lowercase token: blue"}],
        "q3": [{"role": "user", "content": "Output only this digit: 7"}],
    }


def _build_client(args: argparse.Namespace) -> RailClient:
    if args.provider == "vllm-server":
        return RailClient.vllm_server(
            base_url=args.base_url,
            api_key=args.api_key,
        )
    return RailClient.vllm(
        model_id=args.model,
        family=args.family,
        batch_flush_size=args.batch_flush_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        metal_memory_fraction=args.metal_memory_fraction,
        device=args.device,
        extra_llm_kwargs={"max_num_seqs": args.max_num_seqs} if args.max_num_seqs is not None else None,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    result_path = output_dir / "results.jsonl"
    per_request_dir = output_dir / "requests"

    client = _build_client(args)
    request_kwargs = {
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
    }
    if args.enable_thinking:
        request_kwargs["enable_thinking"] = True
    items = batch_items_from_queries(build_queries(), **request_kwargs)
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
        f"{len(items)} provider={args.provider} model={args.model} family={args.family} "
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
