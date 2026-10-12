# tokenrail

[![CI](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml/badge.svg)](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenrail?cacheSeconds=300)](https://pypi.org/project/tokenrail/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/tokenrail/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`tokenrail` is a small Python library for running OpenAI Responses API jobs with a `client.responses`-style surface.

It focuses on:

- thread-based OpenAI batch execution
- structured output parsing for Pydantic models
- client-side RPM / TPM submit throttling
- OpenAI prompt-cache settings passed through unchanged, with read/write usage tracking
- per-model token / cost monitoring with ETA progress reporting
- resumable JSONL and per-request result writing

Fully typed (PEP 561), supports Python 3.10+.

The library is in beta. The public API is still evolving, and breaking changes
may occur between 0.x minor versions.

## Installation

```bash
uv add tokenrail
# or
pip install tokenrail
```

To track an unreleased revision instead, depend on the Git repository directly:

```toml
[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", branch = "main" }
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

## Prompt caching

tokenrail forwards OpenAI's prompt-cache settings and input content as supplied.
It does not detect shared prefixes, insert breakpoints, generate or split keys,
prewarm caches, or impose cache-specific submit limits. These settings work with
both `responses.create(...)` and `responses.parse(...)`, including batch items.

| Setting | Purpose |
| --- | --- |
| `prompt_cache_options` | Cache mode, TTL, prewarming, and diagnostics where supported |
| `prompt_cache_breakpoint` inside `input` content | The end of a reusable prefix |
| `prompt_cache_key` | A caller-supplied key, preserved exactly |
| `prompt_cache_retention` | Retention for older models that support it |

Omitting these settings keeps the OpenAI API defaults, including implicit
caching. On GPT-5.6 and later, writing an eligible prefix costs 1.25 times the
ordinary input rate, so a prefix that is never reused can cost more than an
uncached input. Setting support and cache eligibility depend on the model.

Use the API defaults:

```python
from tokenrail import RailClient

client = RailClient.openai()
response = client.responses.create(model="gpt-6.1-sol", input="Summarize this paper.")
```

Cache only a reusable prefix on GPT-5.6 and later. Put the shared content in its
own block and mark its end; content after the breakpoint stays outside the cache
write. The prefix must render to at least 1,024 visible input tokens. Top-level
`instructions` cannot contain a breakpoint, so supply reusable instructions in
a developer content block:

```python
from tokenrail import BatchExecutor, batch_items_from_queries

shared_block = {
    "type": "input_text",
    "text": "Shared rubric, examples, and reference material ...",  # At least 1,024 tokens.
    "prompt_cache_breakpoint": {"mode": "explicit"},
}
items = batch_items_from_queries(
    {
        item_id: [
            {"role": "developer", "content": [shared_block]},
            {"role": "user", "content": question},
        ]
        for item_id, question in {
            "paper-1": "Summarize paper one.",
            "paper-2": "Summarize paper two.",
        }.items()
    },
    model="gpt-6.1-sol",
    prompt_cache_options={"mode": "explicit"},
)
stats = BatchExecutor(client=client, max_workers=16, max_rpm=120).run(items)
print(stats.cached_tokens)       # Tokens read from cache.
print(stats.cache_write_tokens)  # Tokens newly written to cache.
```

Avoid prompt-cache reads and writes on GPT-5.6 and later by using explicit mode
with no breakpoints:

```python
response = client.responses.create(
    model="gpt-6.1-sol",
    input="A one-off prompt ...",
    prompt_cache_options={"mode": "explicit"},
)
```

Choose keys, breakpoints, and reuse according to your workload. When supplying a
key, keep it stable across related requests. tokenrail records cache read/write
tokens and estimates their costs; it does not guarantee a hit or a saving.
Conflicting values supplied in normal request arguments and `extra_body` raise
`ValueError` before sending. `store=False` controls response storage, not prompt
caching. See the [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
for current model support, pricing, and cache lifetime.

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
- Omitting `timeout` (or passing `None`) preserves the SDK's default timeout. Pass a number of seconds to override it.
- `max_rpm` and `max_tpm` are optional client-side submit limits. When a limit is set, `BatchExecutor` waits before submitting more work instead of raising its effective concurrency above the configured rate.
- TPM throttling estimates future usage from completed requests; it is not a hard API token limit. SDK retries are not separately counted by the submit limiter.
- Request failures are captured as error records (written to sinks and counted in stats) rather than raised, so one failing item does not abort the batch.
- API responses with a status other than `completed`, or with an API error, are counted as errors. Incomplete responses retain partial output and usage for diagnostics. `background=True` is rejected before sending because tokenrail does not poll background responses.
- With the OpenAI SDK, structured-output parsing failures return a response with `error` set and `output_parsed=None`, preserving the raw response, partial text, usage, and estimated cost. Check `response.error` before using parsed output. Capturing the raw response does not issue another API request; transport errors and invalid request arguments still raise on direct client calls.
- `base_url` is passed through to the OpenAI Python SDK for callers that need an SDK-level custom endpoint.

## Resume behavior

`BatchExecutor` reads successfully completed ids from every configured sink
before it starts. An id is skipped only when a valid, successful record exists
in all sinks. Failed, unfinished, and corrupt records are retried on re-running
the batch. Legacy records without a `status` field remain readable.

If saving to a later sink fails, `run()` raises the persistence error. On the
next run, ids missing from any sink are requested again and saved to every
sink. This can incur another API charge and append another JSONL record for
the same id; consumers should use the latest successful record for that id.
Resume does not provide exactly-once API execution. Use the same ids only for
the same inputs and request settings.

`PerRequestJsonSink` writes to a temporary file and atomically replaces the
destination, without following existing destination symlinks. It validates
saved JSON before considering a file complete. Ids must be safe single file
names: path separators, Windows-reserved names, and unsafe filename characters
are rejected before batch requests are sent. `ResultsJsonlSink` repairs a
broken final line before appending and preserves complete final records that
lack a newline. Built-in sinks coordinate threads within one sink instance;
do not concurrently write the same destination from separate instances or
processes.

Successful per-request files contain the raw provider response when available.
Failed requests, including local structured-output parsing failures, contain
the normalized response with `error`, usage, cost, and the original provider
body under `raw_response` when available. This keeps parsing failures
retryable even when the API returned `status="completed"`.

Batch ids must not be empty. If an item supplies `request_id`, it must equal
`BatchItem.id`; empty batch ids and mismatches raise `ValueError` before any
requests in that batch are sent, regardless of the configured sinks.

If you use a custom `projector` with `ResultsJsonlSink`, make sure it keeps an `"id"` field — resume relies on it.
The sink always adds the response's authoritative `error` and `status` fields,
including with custom projectors. Custom sinks should return only valid,
successfully completed ids from `load_done_ids()`.

## Progress output

`RollingMetricsMonitor` keeps request-specific details short and prints batch
metrics separately:

```text
tokenrail · 100 requests
   PAYER: openai — costs are covered
  0001  ok   req-0001      model=gpt-5.6-terra  1.3k tok (40% cached / 0% cache-write)  $0.002000  oai  1.4s
── 50/100 · 50% · 00:00:14 · ETA 00:00:14 · 58 rpm · 74k tpm · $0.100 (oai 100% / dev $0.000) · cache r40%/w2%
!! PAYER SWITCH: openai → developer at 53/100 (00:00:15) — now billed to you
  0053  ok   req-0053      1.2k tok (38% cached / 0% cache-write)  $0.001900  DEV  1.1s
Done 100/100 · 99 ok / 1 errors · 00:00:29
Total $0.198 — openai $0.104 (53%) / developer $0.094 (47%)
Prompt cache: 48k read / 2.4k written
Payer switches: 1
```

Periodic summaries are emitted every 50 requests or 30 seconds by default.
Configure them with `summary_every` and `summary_interval`, adjust payer
transition hysteresis with `payer_switch_threshold`, and control ANSI styling
with `color`. Pass `verbose=True` to use the legacy `[n/total] id=...` format.
`printer=None` still disables all monitor output.

Request lines show `oai` for `openai`, `DEV` for `developer`, and the original
string for any other payer value. A missing or empty payer is shown as `?`.
The monitor reads the payer from `cost`, falling back to `billing.payer` when
unavailable. Unknown payer values count toward `unknown_payer_requests` but
do not drive payer transitions.

## Cost tracking

Standard-tier base prices in USD per million tokens, checked against the
[official OpenAI pricing page](https://developers.openai.com/api/docs/pricing)
on October 8, 2026:

| Model | Input | Cached input | Cache writes | Output |
| --- | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | $10.00 | $1.00 | $12.50 | $50.00 |
| `gpt-6.1-sol` | $2.00 | $0.10 | $2.50 | $10.00 |
| `gpt-6-sol` | $2.00 | $0.20 | $2.50 | $10.00 |
| `gpt-6-luna` | $0.10 | $0.01 | $0.125 | $0.50 |
| `gpt-5.6-sol` / `gpt-5.6` | $4.00 | $0.40 | $5.00 | $20.00 |
| `gpt-5.6-terra` | $2.00 | $0.20 | $2.50 | $12.00 |
| `gpt-5.6-luna` | $0.20 | $0.02 | $0.25 | $1.20 |

GPT-5.6 Sol uses the current promotional rate, available at least through
November 21, 2026 according to OpenAI.

- GPT-5.6 and GPT-6 cache writes are estimated at 1.25 times each model's ordinary input rate and are tracked separately from cache reads.
- Costs use the checked-in base, standard-tier pricing table (`tokenrail.catalog`). The estimate does not apply GPT-5.6/GPT-6 long-context rates above 272K input tokens, Batch/Fast/Flex/Ultrafast pricing, or regional uplifts. The official pricing page is authoritative.
- Models without a pricing entry get `cost=None`. If an unregistered model partially matches an older catalog entry, tokenrail emits `ModelCatalogFallbackWarning` once per model and names the capability and pricing entries used as fallbacks.
- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- `reasoning_effort` is gated to supported `gpt-5`, `gpt-6`, and `o`-series models in the checked-in capability registry.
- GPT-6 Astra and GPT-6.1 Sol support `low`, `medium`, `high`, `xhigh`, and `max` reasoning efforts; `temperature` and `top_p` are rejected before sending a request. GPT-6 Sol and Luna additionally support `none`, which must be explicitly selected to use `temperature` or `top_p`. These checks apply to both `responses.create(...)` and `responses.parse(...)`. See the [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model).

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

Live API checks are opt-in and spend tokens. Set `OPENAI_API_KEY` in the
environment or a local `.env`, then run:

```bash
TOKENRAIL_RUN_LIVE_TESTS=1 uv run pytest tests/test_live_openai.py -q -s
```

These checks use `gpt-6-luna`, capped outputs, and no SDK retries. The normal
test command skips them. The suite records usage before SDK parsing and checks
a 100,000-token ceiling; its fixed prompts and output caps keep the expected
usage well below that ceiling. During 0.4.1 release validation, the five live
checks passed using 7,719 tokens over ten requests, including 3,604 cache reads
and 3,604 cache writes.

## License

[MIT](LICENSE)
