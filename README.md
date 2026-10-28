# tokenrail

[![CI](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml/badge.svg)](https://github.com/takumi0shibata/tokenrail/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenrail?cacheSeconds=300)](https://pypi.org/project/tokenrail/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/tokenrail/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`tokenrail` is a small Python library for running OpenAI Responses and Decisions
API jobs with SDK-style `client.responses` and `client.decisions` namespaces.

It focuses on:

- thread-based OpenAI and Azure OpenAI batch execution
- structured output parsing for Pydantic models
- client-side RPM / TPM submit throttling
- OpenAI prompt-cache settings passed through unchanged, with read/write usage tracking
- per-model token / cost monitoring with ETA progress reporting
- resumable JSONL result writing with the full API response

The package includes type annotations and PEP 561 metadata. It requires Python
3.10+ and OpenAI Python SDK 3.26.0 or later (installed automatically). You need an OpenAI
API key and access to the model you request. API calls can incur charges.

Batches run locally in threads using the Responses or Decisions API; this is not a wrapper
for OpenAI’s hosted Batch API.

The library is in beta. The public API is still evolving, and breaking changes
may occur between 0.x minor versions.

## Installation

```bash
uv add tokenrail
# or
pip install tokenrail
```

Set your API key in the environment before running Python:

```bash
export OPENAI_API_KEY="your-api-key"
```

The library does not load `.env` files automatically. If your application uses
one, load it before constructing the client. Do not commit API keys or result
files containing sensitive prompts and responses.

## Quick start

Send one request and inspect its result:

```python
from tokenrail import RailClient

client = RailClient.openai(timeout=60, max_retries=2)
response = client.responses.create(
    model="gpt-6-luna",
    input="Explain what a prime number is in one sentence.",
    reasoning_effort="none",
    max_output_tokens=128,
)
if response.error:
    raise RuntimeError(response.error)
print(response.output_text)
print(response.usage.total_tokens)
```

Use a model available to your account. `responses.create()` returns a
`NormalizedResponse`, not the SDK response object. Transport and request errors
raise exceptions; an API response that is incomplete or reports an error is
returned with `response.error` set. Inspect that field before using the output.

## Run a batch

This example is independent of the quick start. Each dictionary key is a stable
job ID; each value is the input for one request.

```python
from tokenrail import BatchExecutor, RailClient, batch_items_from_queries

client = RailClient.openai(timeout=60, max_retries=2)
items = batch_items_from_queries(
    {
        "prime": "Explain a prime number in one sentence.",
        "factor": "Explain a factor in one sentence.",
    },
    model="gpt-6-luna",
    reasoning_effort="none",
    max_output_tokens=128,
)
stats = BatchExecutor(
    client=client,
    output_path="out/results.jsonl",
    max_workers=2,
    max_rpm=30,
).run(items)
print(stats.to_dict())
```

Choose concurrency and rate limits for your account. Run the same batch again
with the same output path to skip saved successes and retry failed or unfinished
jobs. Use a new path when changing inputs or settings; see [Resume behavior](#resume-behavior).
The result-saving examples below reuse `client` and `items` from this example.

## Run Decisions jobs

Include `questions` to run a batch through the Decisions API. No API selector is
needed: each item's request determines its operation. `questions` selects
`decisions.create`; otherwise a non-null `text_format` selects `responses.parse`,
and other items use `responses.create`.

```python
from tokenrail import BatchExecutor, RailClient, batch_items_from_queries

client = RailClient.openai(timeout=60, max_retries=2)
decision_items = batch_items_from_queries(
    {
        "ticket-1": "I was charged twice. Please refund the extra charge.",
        "ticket-2": "Where is my order?",
    },
    model="gpt-6-luna",
    questions=[{
        "type": "predicate",
        "name": "refund_requested",
        "instructions": "Does the customer request a refund?",
    }],
)
stats = BatchExecutor(
    client=client,
    output_path="out/decisions.jsonl",
    max_workers=10,
    max_rpm=60,
).run(decision_items)
```

For a single request, call `client.decisions.create(model=..., input=...,
questions=...)`. It returns a `NormalizedResponse` with `answers`, a list of
dictionaries in question order. The answer types are `predicate`, `choice`,
`score`, and `refusal`; inspect each answer's `type` before using its fields.
An individual question's refusal is saved as an answer and does not cause an
automatic retry of the request. `output_text` and `output_parsed` are unused
for Decisions; the original response is preserved in `raw_response`.

`batch_items_from_queries()` consumes shared question iterables once, including
`choices`, `levels`, and instruction blocks supported by the installed SDK. The
generated jobs retain that content for later jobs and resume without changing
the caller's dictionaries. String instructions stay unchanged. This does not
rewind an exhausted iterator reused in separate direct calls or raw `BatchItem`
objects.

Text, image and JSON content, question definitions, and prompt-cache settings
are forwarded as supplied. `prompt_cache_key` and `prompt_cache_options` work
across supported SDK versions. `request_id` identifies the local job and
`metadata` is saved locally for Decisions; neither is sent to that API.
See [Decisions](https://developers.openai.com/api/docs/guides/decisions) for
accepted inputs, supported models and question types.

To mix Decisions, parsed output, and ordinary Responses in one batch, build
individual `BatchItem` objects with their own `request_kwargs`. A supplied
`questions` value must contain 1–400 question dictionaries; `None`, an empty
collection, or a combination with `text_format` raises `ValueError` before any
request in the batch is sent. `questions` supplied in `extra_body` also selects
Decisions. Direct calls through `client.responses` reject `questions`; use
`client.decisions.create` explicitly for single requests.

Concurrency and RPM/TPM submit limits are shared across all items in a mixed
batch. Each request log and saved record includes its operation as `api`, such
as `decisions.create` or `responses.parse`. Resume still uses job IDs, so use a
new output path when changing the API, questions, inputs or other settings.

### Azure OpenAI

Use Azure's v1 endpoint with the OpenAI SDK. `model` is your deployment name.
The same configuration supports Decisions and Responses jobs.

```python
import os
from openai import OpenAI
from tokenrail import RailClient

with OpenAI(
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    timeout=60,
    max_retries=2,
) as sdk:
    azure = RailClient.openai(client=sdk)
    result = azure.decisions.create(
        model="YOUR-GPT-6-LUNA-DEPLOYMENT",
        input="Please refund the duplicate charge.",
        questions=[{
            "type": "predicate",
            "name": "refund_requested",
            "instructions": "Does the customer request a refund?",
        }],
    )
    print(result.answers)
```

For Microsoft Entra ID authentication, configure the SDK client with your
token provider before injecting it. Confirm that your deployment and region
support Decisions; see the
[Azure guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/decisions).
Decisions usage is tracked on Azure and other custom endpoints, but `cost` is
`None`: tokenrail does not assume OpenAI prices apply to them.

## Result saving

`output_path` is required for every batch. Parent directories are created
if needed. Each completed request is appended as one JSONL record, in
completion order rather than input order.

By default, records contain the full normalized response: output text, parsed
output, refusal, Decisions answers, API operation, usage, billing, estimated cost, timing, metadata, status, and
error. The original API response body is preserved under `raw_response` when
available. The top-level `id` is your query id; the API's response id is kept
under `raw_response.id`.

To save selected fields, pass a `projector` directly:

```python
executor = BatchExecutor(
    client=client,
    output_path="out/selected-results.jsonl",
    projector=lambda response: {
        "text": response.output_text,
        "usage": response.usage.to_dict(),
    },
)
stats = executor.run(items)
```

The projector receives a `NormalizedResponse` and must return a dictionary
with JSON-serializable values. tokenrail always adds or overrides `id`, `error`,
`status`, and `api` with the actual result, so field selection cannot break resume or
hide failures. A projector replaces the default record; unselected fields,
including `raw_response`, are omitted.

## Structured output

Use `responses.parse()` with a Pydantic model to validate structured output:

```python
from pydantic import BaseModel

from tokenrail import RailClient


class Summary(BaseModel):
    topic: str
    main_point: str


client = RailClient.openai(timeout=60, max_retries=2)
response = client.responses.parse(
    model="gpt-6-luna",
    input="Summarize: A prime number is an integer greater than one with exactly two positive divisors.",
    reasoning_effort="none",
    max_output_tokens=256,
    text_format=Summary,
)
if response.error:
    raise RuntimeError(response.error)
if response.refusal:
    print(response.refusal)
elif response.output_parsed is not None:
    print(response.output_parsed.model_dump())
```

For structured batches, pass `text_format=Summary` to
`batch_items_from_queries()` or include it in each `BatchItem.request_kwargs`.
The executor selects `responses.parse()` for those items. Saved default records
include the parsed output in JSON form.

Use `response_format={...}` with `responses.create()` to supply a raw JSON
Schema without Pydantic parsing. Do not combine `response_format` with
`text_format`, or supply `verbosity` with `text_format`; these combinations
are rejected before sending.

## Prompt caching

tokenrail forwards cache settings and input content to OpenAI. It does not
insert breakpoints, generate keys, prewarm caches automatically, or guarantee
cache hits or savings. Model support, cache lifetime and prices are determined
by the API; consult the [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

| Setting | Passed through as |
| --- | --- |
| `prompt_cache_key` | A caller-supplied key, preserved exactly |
| `prompt_cache_options` | Cache options supported by the selected model |
| `prompt_cache_retention` | Retention for models that support it |
| `prompt_cache_breakpoint` | A marker inside an input content block |

These settings work with create, parse and batch requests. Omitting them keeps
the API defaults. Keep keys stable for related requests and supply reusable
content according to the model’s caching requirements. `store=False` controls
response storage, not prompt caching.

Cache reads and writes are reported separately as `cached_tokens` and
`cache_write_tokens` in usage and batch statistics. Conflicting values in normal
arguments and `extra_body` raise `ValueError` before sending.

## Configuration and API reference

| API | Purpose |
| --- | --- |
| `RailClient.openai(...)` | Construct a client or wrap an existing OpenAI SDK client with `client=...` |
| `client.responses.create(...)` | Return a normalized text or JSON response |
| `client.responses.parse(..., text_format=...)` | Return validated Pydantic output |
| `client.decisions.create(..., questions=...)` | Return typed Decisions answers, usage and cost estimate |
| `batch_items_from_queries(queries, **kwargs)` | Apply shared request settings to an ID-to-input mapping |
| `BatchItem(id, request_kwargs)` | Supply different request settings for individual jobs |
| `BatchExecutor(client=..., output_path=..., ...)` | Run and persist jobs; `run(items)` returns `StatsSnapshot` |

`RailClient.openai()` accepts `api_key`, `organization`, `base_url`, `timeout`
and `max_retries`. The default retry count is 2. For explicit connection cleanup,
create an OpenAI SDK client in its context manager and inject it with `client=`;
configure credentials, timeout and retries on that injected client.


- `max_retries` configures the OpenAI Python SDK client's built-in retry behavior. `tokenrail` does not add its own retry loop on top.
- Omitting `timeout` (or passing `None`) preserves the SDK's default timeout. Pass a number of seconds to override it.
- `max_rpm` and `max_tpm` are optional client-side submit limits. When a limit is set, `BatchExecutor` waits before submitting more work instead of raising its effective concurrency above the configured rate.
- TPM throttling estimates future usage from completed requests; it is not a hard API token limit. SDK retries are not separately counted by the submit limiter.
- Request failures are captured as error records (saved to the JSONL file and counted in stats) rather than raised, so one failing item does not abort the batch.
- API responses with a status other than `completed`, or with an API error, are counted as errors. Incomplete responses retain partial output and usage for diagnostics. `background=True` is rejected before sending because tokenrail does not poll background responses.
- `stream=True` is rejected before sending, including through `extra_body`, for both `responses.create(...)` and `responses.parse(...)`. tokenrail requires non-streaming responses to retain output and usage; omit `stream` or set it to `False`.
- With the OpenAI SDK, structured-output parsing failures return a response with `error` set and `output_parsed=None`, preserving the raw response, partial text, usage, and estimated cost. Check `response.error` before using parsed output. Capturing the raw response does not issue another API request; transport errors and invalid request arguments still raise on direct client calls.
- `base_url` is passed through to the OpenAI Python SDK for callers that need an SDK-level custom endpoint.

## Resume behavior

`BatchExecutor` reads successful query ids from `output_path` before it starts.
An id is skipped when a valid record has no error and its status is `completed`
(or absent in legacy records). Failed, unfinished, and corrupt records are
retried on re-running the batch. Existing JSONL files from earlier releases
remain readable.

Use the same output path and query ids only when resuming the same inputs and
request settings. Select a different path for a different batch. Resume does
not provide exactly-once API execution: an interruption can cause another API
charge and another record for the same id. Consumers should use the latest
successful record for each id.

Projector, JSON serialization, and file-writing errors propagate from `run()`.
Results that were not saved successfully remain eligible for retry. JSONL
writes are flushed and synced to disk. A broken final line is repaired before
appending; complete final records without a newline are preserved. Writes are
coordinated within one executor; do not concurrently write the same destination
from separate executors or processes.

The default record preserves partial output, usage, cost, and `raw_response`
for failed requests, including local structured-output parsing failures.
These failures remain retryable even when the API returned `status="completed"`,
because the top-level `error` records the parsing failure.

Batch ids must be non-empty and unique after conversion to strings. If an item
supplies `request_id`, it must equal `BatchItem.id`. Empty ids, duplicate ids,
and mismatches raise `ValueError` before any requests in that batch are sent.
Query ids are stored as JSON values and are not used as file names.

## Progress output

`BatchExecutor` automatically tracks statistics, keeps request-specific
details short, and prints batch metrics separately:

```text
tokenrail · 100 requests
   PAYER: openai — costs are covered
  0001  ok   req-0001      api=responses.create  model=gpt-5.6-terra  1.3k tok (40% cached / 0% cache-write)  $0.002000  oai  1.4s
── 50/100 · 50% · 00:00:14 · ETA 00:00:14 · 58 rpm · 74k tpm · $0.100 (oai 100% / dev $0.000) · cache r40%/w2%
!! PAYER SWITCH: openai → developer at 53/100 (00:00:15) — now billed to you
  0053  ok   req-0053      api=responses.create  1.2k tok (38% cached / 0% cache-write)  $0.001900  DEV  1.1s
Done 100/100 · 99 ok / 1 errors · 00:00:29
Total $0.198 — openai $0.104 (53%) / developer $0.094 (47%)
Prompt cache: 48k read / 2.4k written
Payer switches: 1
```

Periodic summaries are emitted every 50 completed requests or after 30 seconds
when another result arrives. Payer transitions require three consecutive known
payer observations, and ANSI colors are enabled automatically for a terminal.

Pass `progress=False` to disable all progress output:

```python
stats = BatchExecutor(
    client=client,
    output_path="out/results.jsonl",
    progress=False,
).run(items)
```

Statistics are still collected and returned as a `StatsSnapshot`. Display
intervals and formatting use built-in defaults.

Request lines show `oai` for `openai`, `DEV` for `developer`, and the original
string for any other payer value. A missing or empty payer is shown as `?`.
The monitor reads the payer from `cost`, falling back to `billing.payer` when
unavailable. Unknown payer values count toward `unknown_payer_requests` but
do not drive payer transitions.

## Cost tracking and model support

`response.usage` records token counts; `response.cost` estimates USD costs using
the checked-in [model catalog](src/tokenrail/catalog.py). These are estimates,
not invoices. Prices can change, and the catalog does not account for all
service tiers, long-context rates or regional adjustments. Refer to
[OpenAI pricing](https://developers.openai.com/api/docs/pricing) for current rates.

Models without a pricing entry return `cost=None`. Unregistered model names
that partially match known entries can use fallback capabilities or prices and
emit `ModelCatalogFallbackWarning`. Check the warning before relying on the
estimate or supported parameters.

Prices are selected per API operation. Decisions has separate rates, including
zero output-token cost and distinct cache-read/write accounting. Its price
estimate is available for registered models on `https://api.openai.com/v1/`;
custom endpoints and Azure deployments return `cost=None`. Usage is always
recorded when supplied by the API. `stats.unpriced_requests` counts results
without a cost estimate. Progress summaries label partial cost totals as a
known subtotal and show `$—` when every result is unpriced. Numeric cost totals
in statistics contain only known estimates, not an estimate for the missing
requests.

The capability catalog validates options such as `reasoning_effort`,
`temperature` and `top_p` before sending. Support is model-specific; for example,
GPT-6 Luna requires `reasoning_effort="none"` when setting sampling parameters.

Where the API includes `billing.payer`, tokenrail uses it to attribute nominal
costs. A payer of `openai` is recorded as OpenAI-covered. Unknown payer values
remain unknown; monitoring does not enforce a spending limit or stop batches
when the payer changes.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing key or authentication error | Set `OPENAI_API_KEY` in the process environment; confirm account and model access. |
| `ValueError` before a request | Check model capabilities, incompatible structured-output options, and batch IDs. |
| Rate-limit errors | Reduce `max_workers`/`max_rpm`/`max_tpm`; account for other workloads and SDK retries. |
| No requests on a repeat run | Successful IDs already exist in the JSONL file. Use a new path for a new workload. |
| Missing parsed output | Inspect `error`, `refusal`, `status` and `raw_response`; incomplete output may require a higher output cap. |
| Missing cost estimate | The model may not have a catalog entry; token usage can still be available. |
| File-writing or projector error | Fix the destination or serialization issue before resuming; unsaved requests may run again. |

Streaming and background polling are unsupported. Resume is not an exactly-once
guarantee, and token throttling is not a monetary budget. See the configuration
and resume sections before running large workloads.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

Live API checks are optional and can incur charges. They are skipped by default.
To run them from a repository checkout, provide `OPENAI_API_KEY` in the environment
or a local `.env`, then explicitly enable them:

```bash
TOKENRAIL_RUN_LIVE_TESTS=1 uv run pytest tests/test_live_openai.py -q -s
```

The test-only transport uses GPT-6 Luna, disables SDK retries and serializes
requests. It reserves counted input plus capped output against a 100,000-token
limit per invocation and checks each response’s payer and usage before allowing
further sends. Reservations are not refunded. Unknown/non-OpenAI payer, invalid
usage or transport failures stop further sends; the request that detects a
payer change may already be charged. This guard is not part of the installed
library. Repeated invocations have separate budgets.

## Related documentation

- [Changelog](CHANGELOG.md) and [releases](https://github.com/takumi0shibata/tokenrail/releases)
- [API types and result fields](src/tokenrail/types.py)
- [Model capabilities and pricing catalog](src/tokenrail/catalog.py)
- [Report a bug](https://github.com/takumi0shibata/tokenrail/issues)

## License

[MIT](LICENSE)
