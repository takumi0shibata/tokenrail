# tokenrail

`tokenrail` is a small Python library for running OpenAI Responses API jobs and local vLLM models with the same `client.responses.create(...)`-style surface.

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
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.2.0" }
```

Then sync:

```bash
uv sync
```

If you want local vLLM support too:

```toml
[project]
dependencies = [
    "tokenrail[vllm]",
]

[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.2.0" }
```

On Linux this installs vLLM. On Apple Silicon macOS, use Python 3.12 or newer; the same extra installs vLLM-Metal, the vLLM hardware plugin for Metal/MLX acceleration.

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

`max_retries` configures the OpenAI Python SDK client's built-in retry behavior. `tokenrail` does not add its own retry loop on top.

## Local vLLM usage

```python
from tokenrail import RailClient

client = RailClient.vllm(
    model_id="Qwen/Qwen3.5-9B",
    family="qwen",
    batch_flush_size=256,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
    trust_remote_code=True,
    seed=12,
)

response = client.responses.create(
    model="Qwen/Qwen3.5-9B",
    input=[{"role": "user", "content": "Write a haiku about caching."}],
    max_output_tokens=64,
    temperature=0.2,
    enable_thinking=True,
)

print(response.output_text)
print(response.usage.to_dict())
```

For Apple Silicon, use vLLM-Metal with an MLX-optimized text model:

```python
from tokenrail import RailClient

client = RailClient.vllm(
    model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    family="qwen",
    dtype="auto",
    max_model_len=2048,
    metal_memory_fraction="auto",
    extra_llm_kwargs={"max_num_seqs": 8},
)
```

`gpu_memory_utilization` is the CUDA vLLM memory knob. On Apple Silicon, use `metal_memory_fraction` or set `VLLM_METAL_MEMORY_FRACTION` directly.

## Local vLLM batch execution

```python
from tokenrail import BatchExecutor, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries

client = RailClient.vllm(
    model_id="Qwen/Qwen3.5-9B",
    family="qwen",
    batch_flush_size=256,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
    trust_remote_code=True,
    seed=12,
)

queries = {
    "1": [{"role": "user", "content": "Summarize this essay in 3 bullets."}],
    "2": [{"role": "user", "content": "Extract the grading rationale."}],
    "3": [{"role": "user", "content": "Give a one-line final judgment."}],
}

items = batch_items_from_queries(
    queries,
    model="Qwen/Qwen3.5-9B",
    max_output_tokens=128,
    enable_thinking=False,
)

executor = BatchExecutor(
    client=client,
    sinks=[
        ResultsJsonlSink("out/results.jsonl"),
        PerRequestJsonSink("out/requests"),
    ],
    monitor=RollingMetricsMonitor(),
)

stats = executor.run(items)
print(stats.to_dict())
```

`BatchExecutor` passes all pending requests to the vLLM provider, and the provider groups them by sampling settings before calling `llm.generate(...)`.
`batch_flush_size` is only the maximum number of prompts per `generate(...)` call for one sampling group; actual continuous batching and GPU scheduling are handled by vLLM itself.

Current built-in families:

- `qwen`: uses the tokenizer chat template with `enable_thinking=...`
- `gemma`: enables thinking by prefixing the system prompt with `<|think|>`
- `batch_flush_size`: controls only how many prompts `tokenrail` passes to one `llm.generate(...)` call; actual scheduling/parallelism stays inside vLLM

## Notes

- Apple Silicon GPU support depends on vLLM-Metal: see the vLLM [Apple Silicon GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [CPU Apple Silicon notes](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/), [vLLM-Metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/), and [supported models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/).
- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- v1 local vLLM support is text-only. Multimodal local inputs can be added behind the same provider interface later.
- `reasoning_effort` is gated to `gpt-5` / `o`-series style models in the checked-in capability registry.
- `reasoning_effort`, `verbosity`, `response_format`, `service_tier`, and `store` are intentionally rejected for local vLLM providers.
