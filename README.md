# tokenrail

[![CI](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml/badge.svg)](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenrail)](https://pypi.org/project/tokenrail/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/tokenrail/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`tokenrail` is a small Python library for running OpenAI Responses API jobs with a `client.responses`-style surface.

It focuses on:

- thread-based OpenAI batch execution
- structured output parsing for Pydantic models
- client-side RPM / TPM submit throttling
- GPT-5.6 explicit prompt caching with deterministic cache-key sharding
- per-model token / cost monitoring with ETA progress reporting
- resumable JSONL and per-request result writing

Fully typed (PEP 561), supports Python 3.10+.

## Installation

```bash
uv add tokenrail
# or
pip install tokenrail
```

To track an unreleased revision instead, depend on the Git repository directly:

```toml
[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v2.0.0" }
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

## Prompt cache auto-sharding

GPT-5.6 and later can cache an exact reusable prompt prefix at an explicit
breakpoint. Enable tokenrail's opt-in planner to find the longest common
message/content-block prefix, mark it for explicit-only caching, and partition
the batch over stable physical cache keys:

```python
from tokenrail import BatchExecutor, PromptCacheConfig, RailClient
from tokenrail.executor import batch_items_from_queries

client = RailClient.openai()

items = batch_items_from_queries(
    {
        "paper-1": "Summarize paper one.",
        "paper-2": "Summarize paper two.",
    },
    model="gpt-5.6-terra",
    # In a real workload this reusable prefix must render to at least 1,024 tokens.
    instructions="Follow the shared rubric and examples: ...",
)

stats = BatchExecutor(
    client=client,
    max_workers=32,
    max_rpm=120,
    prompt_cache="auto",
).run(items)

print(stats.prompt_cache_shards)       # 8
print(stats.cached_tokens)             # tokens read from cache
print(stats.cache_write_tokens)        # tokens newly written to cache
```

Auto mode computes `ceil(max_rpm / 15)`, maps each stable `BatchItem.id` to a
shard with SHA-256, and enforces a rolling 15 RPM submit limit on every
physical key. To supply a logical key, override the RPM estimate, or set the
shard count directly, pass a configuration object:

```python
prompt_cache = PromptCacheConfig(
    base_key="paper-summary-v3",
    shards=8,                    # "auto" by default
    expected_rpm=None,           # falls back to BatchExecutor.max_rpm
    target_rpm_per_shard=15,
)
```

The planner deliberately works at content-block boundaries, not in the middle
of strings. A batch must use one OpenAI GPT-5.6 model and one shared request
configuration. Stateful inputs such as `previous_response_id`, batches without
a common cacheable block, and inputs that already contain explicit cache
configuration are rejected before any API request is sent. Shared top-level
`instructions` are moved to a leading developer `input_text` block so the
prefix can be marked explicitly.

The OpenAI service still requires an exact prefix of at least 1,024 tokens.
Sharding improves cache routing; it does not relax ordinary API RPM/TPM limits,
and excess shards duplicate the initial cache write. See the
[OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
for the authoritative behavior.

## Structured output batches

Pass a Pydantic model as `text_format` when building batch items. `BatchExecutor`
will call `responses.parse(...)` for those items and store the validated object
on `response.output_parsed`.

```python
from pydantic import BaseModel

from tokenrail import BatchExecutor, RailClient, ResultsJsonlSink
from tokenrail.executor import batch_items_from_queries


class PaperSummary(BaseModel):
    title: str
    key_assumptions: list[str]


client = RailClient.openai(max_retries=6)

items = batch_items_from_queries(
    {
        "paper-1": [{"role": "user", "content": "Extract the title and assumptions from this paper: ..."}],
        "paper-2": [{"role": "user", "content": "Extract the title and assumptions from this paper: ..."}],
    },
    model="gpt-5.4-mini-2026-03-17",
    reasoning_effort="medium",
    text_format=PaperSummary,
)

sink = ResultsJsonlSink(
    "out/structured-results.jsonl",
    projector=lambda response: {
        "id": response.id,
        "summary": response.output_parsed.model_dump(mode="json") if response.output_parsed else None,
        "refusal": response.refusal,
        "usage": response.usage.to_dict(),
    },
)

stats = BatchExecutor(client=client, sinks=[sink], max_workers=16).run(items)
print(stats.to_dict())
```

Use `response_format={...}` with `client.responses.create(...)` when you want to
provide a raw JSON Schema yourself. Use `text_format=YourModel` for Pydantic
parsing; `response_format` and `text_format` cannot be used together. Do not set
`verbosity` on structured output batches that use `text_format`, because the
OpenAI SDK's `responses.parse(...)` path does not accept that combination.

## Configuration notes

- `max_retries` configures the OpenAI Python SDK client's built-in retry behavior. `tokenrail` does not add its own retry loop on top.
- `max_rpm` and `max_tpm` are optional client-side submit limits. When a limit is set, `BatchExecutor` waits before submitting more work instead of raising its effective concurrency above the configured rate.
- Request failures are captured as error records (written to sinks and counted in stats) rather than raised, so one failing item does not abort the batch.
- `base_url` is passed through to the OpenAI Python SDK for callers that need an SDK-level custom endpoint.

## Resume behavior

`BatchExecutor` reads completed ids from the first configured sink before it starts. Re-running the same job with the same output path skips records that are already present, then writes only the remaining requests.

If you use a custom `projector` with `ResultsJsonlSink`, make sure it keeps an `"id"` field — resume relies on it.

## Progress output

`RollingMetricsMonitor` keeps request-specific details short and prints batch
metrics separately:

```text
tokenrail · 100 requests · prompt cache 7 shards @ 15 rpm/key
   PAYER: openai — costs are covered
  0001  ok   req-0001      model=gpt-5.6-terra  1.3k tok (40% cached / 0% cache-write)  $0.002000  oai  1.4s
── 50/100 · 50% · 00:00:14 · ETA 00:00:14 · 58 rpm · 74k tpm · $0.100 (oai 100% / dev $0.000) · cache r40%/w2%
!! PAYER SWITCH: openai → developer at 53/100 (00:00:15) — now billed to you
  0053  ok   req-0053      1.2k tok (38% cached / 0% cache-write)  $0.001900  DEV  1.1s
Done 100/100 · 99 ok / 1 errors · 00:00:29
Total $0.198 — openai $0.104 (53%) / developer $0.094 (47%)
Prompt cache: 7 shards · 48k read / 2.4k written
Payer switches: 1
```

Periodic summaries are emitted every 50 requests or 30 seconds by default.
Configure them with `summary_every` and `summary_interval`, adjust payer
transition hysteresis with `payer_switch_threshold`, and control ANSI styling
with `color`. Pass `verbose=True` to use the legacy `[n/total] id=...` format.
`printer=None` still disables all monitor output.

## Cost tracking

- GPT-5.6 Sol (`gpt-5.6-sol` and the `gpt-5.6` alias) is estimated at $5.00 input / $0.50 cached input / $30.00 output per million tokens.
- GPT-5.6 Terra (`gpt-5.6-terra`) is estimated at $2.50 / $0.25 / $15.00, and GPT-5.6 Luna (`gpt-5.6-luna`) at $1.00 / $0.10 / $6.00.
- GPT-5.6 cache writes are estimated at 1.25 times each model's ordinary input rate and are tracked separately from cache reads.
- Costs use the checked-in base, standard-tier pricing table (`tokenrail.catalog`). The estimate does not apply GPT-5.6 long-context rates above 272K input tokens, Batch/Fast/Flex pricing, or regional uplifts. The [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing) is authoritative.
- Models without a pricing entry get `cost=None`. If an unregistered model partially matches an older catalog entry, tokenrail emits `ModelCatalogFallbackWarning` once per model and names the capability and pricing entries used as fallbacks.
- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- `reasoning_effort` is gated to `gpt-5` / `o`-series style models in the checked-in capability registry.

Fallback warnings can be filtered with Python's standard warning controls:

```python
import warnings

from tokenrail.catalog import ModelCatalogFallbackWarning

warnings.filterwarnings("ignore", category=ModelCatalogFallbackWarning)
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## License

[MIT](LICENSE)
