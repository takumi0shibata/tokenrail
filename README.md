# tokenrail

`tokenrail` is a small Python library for running OpenAI Responses API jobs and local Hugging Face models with the same `client.responses.create(...)`-style surface.

It focuses on:

- thread-based OpenAI batch execution
- per-model token / TPM / RPM / cost monitoring

## Install From GitHub With uv

Add `tokenrail` as a Git dependency from your own project.

```toml
[project]
dependencies = [
    "tokenrail",
]

[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.1.1" }
```

Then sync:

```bash
uv sync
```

If you want Hugging Face local-model support too:

```toml
[project]
dependencies = [
    "tokenrail[hf]",
]

[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.1.1" }
```

Set your API key in the consuming project before using OpenAI:

```bash
export OPENAI_API_KEY=...
```

## OpenAI usage

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
    sinks=[result_sink, per_request_sink],
    monitor=RollingMetricsMonitor(),
)

stats = executor.run(items)
print(stats.to_dict())
```

## Local Hugging Face usage

```python
from tokenrail import RailClient

client = RailClient.hf(
    model_id="Qwen/Qwen3-0.6B-Instruct",
    batch_size=8,
)

response = client.responses.create(
    model="Qwen/Qwen3-0.6B-Instruct",
    input=[{"role": "user", "content": "Write a haiku about caching."}],
    max_output_tokens=64,
    temperature=0.2,
)

print(response.output_text)
print(response.usage.to_dict())
```

## Notes

- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- v1 local HF support is text-only. Multimodal local inputs can be added behind the same provider interface later.
- `reasoning_effort` is gated to `gpt-5` / `o`-series style models in the checked-in capability registry.
