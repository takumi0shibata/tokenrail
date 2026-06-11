# tokenrail

[![CI](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml/badge.svg)](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/takumi0shibata/tokenrail)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`tokenrail` is a small Python library for running OpenAI Responses API jobs with a `client.responses.create(...)`-style surface.

It focuses on:

- thread-based OpenAI batch execution
- client-side RPM / TPM submit throttling
- per-model token / cost monitoring with ETA progress reporting
- resumable JSONL and per-request result writing

Fully typed (PEP 561), supports Python 3.10+.

## Installation

Add `tokenrail` as a Git dependency from your own project:

```toml
[project]
dependencies = [
    "tokenrail",
]

[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v1.0.0" }
```

Then sync:

```bash
uv sync
```

Set your OpenAI API key in the consuming project before use:

```bash
export OPENAI_API_KEY=...
```

## Quick start

```python
from tokenrail import BatchExecutor, ResultsJsonlSink, PerRequestJsonSink, RailClient, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries

client = RailClient.openai(max_retries=6)

queries = {
    "1": [{"role": "user", "content": "Summarize this paper in 3 bullets."}],
    "2": [{"role": "user", "content": "Extract the key assumptions."}],
}

items = batch_items_from_queries(
    queries,
    model="gpt-5.4-mini-2026-03-17",
    reasoning_effort="medium",
    verbosity="low",
)

# Consolidate only the necessary elements from all processing results into a single file.
result_sink = ResultsJsonlSink(
    "out/results.jsonl",
    projector=lambda response: {
        "id": response.id,
        "text": response.output_text,
        "model": response.model,
        "usage": response.usage.to_dict(),
    },
)
# Save the raw output of each query.
per_request_sink = PerRequestJsonSink("out/")

executor = BatchExecutor(
    client=client,
    max_workers=16,
    max_rpm=500,
    max_tpm=200_000,
    sinks=[result_sink, per_request_sink],
    monitor=RollingMetricsMonitor(),
)

stats = executor.run(items)
print(stats.to_dict())
```

## Configuration notes

- `max_retries` configures the OpenAI Python SDK client's built-in retry behavior. `tokenrail` does not add its own retry loop on top.
- `max_rpm` and `max_tpm` are optional client-side submit limits. When a limit is set, `BatchExecutor` waits before submitting more work instead of raising its effective concurrency above the configured rate.
- Request failures are captured as error records (written to sinks and counted in stats) rather than raised, so one failing item does not abort the batch.
- `base_url` is passed through to the OpenAI Python SDK for callers that need an SDK-level custom endpoint.

## Resume behavior

`BatchExecutor` reads completed ids from the first configured sink before it starts. Re-running the same job with the same output path skips records that are already present, then writes only the remaining requests.

If you use a custom `projector` with `ResultsJsonlSink`, make sure it keeps an `"id"` field — resume relies on it.

## Cost tracking

- Costs are estimated from a checked-in per-model pricing table (`tokenrail.catalog`). Models without a pricing entry get `cost=None`; prices may lag behind OpenAI's official pricing page, which is always authoritative.
- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- `reasoning_effort` is gated to `gpt-5` / `o`-series style models in the checked-in capability registry.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## License

[MIT](LICENSE)
