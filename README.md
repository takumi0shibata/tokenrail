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
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.2.1" }
```

Then sync:

```bash
uv sync
```

If you want in-process local vLLM support on Linux too:

```toml
[project]
dependencies = [
    "tokenrail[vllm]",
]

[tool.uv.sources]
tokenrail = { git = "https://github.com/takumi0shibata/tokenrail", tag = "v0.2.1" }
```

On Linux this installs vLLM for in-process execution. On Apple Silicon macOS, use vLLM-Metal as an OpenAI-compatible server instead of installing `tokenrail[vllm]`; the normal vLLM package pulls CUDA/NVIDIA dependencies that are not installable on macOS.

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

## Local vLLM Usage On Linux

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

## Local vLLM-Metal Usage On Apple Silicon

Install vLLM-Metal using the official installer. It creates a separate `~/.venv-vllm-metal` environment containing vLLM-Metal, vLLM core, MLX, and related libraries. Keep this environment separate from the project environment that runs tokenrail.

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Start the vLLM-Metal server in one shell:

```bash
~/.venv-vllm-metal/bin/vllm serve \
  mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 512 \
  --max-num-seqs 8
```

Run tokenrail from another shell. On Apple Silicon, the `sample` command defaults to `--provider vllm-server` and connects to `http://localhost:8000/v1`.

```bash
uv sync
uv run sample \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --max-output-tokens 16 \
  --output-dir sample_out_metal
```

Expected smoke-test result is `success_requests: 3` and `error_requests: 0`. If the server is not running or sandboxed networking blocks localhost, the result records will contain `APIConnectionError`.

In another shell, connect tokenrail to the vLLM-Metal OpenAI-compatible server:

```python
from tokenrail import RailClient

client = RailClient.vllm_server(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.responses.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    input=[{"role": "user", "content": "Reply with exactly: ok"}],
    max_output_tokens=16,
)

print(response.output_text)
```

Batch execution uses the same `BatchExecutor` surface. `tokenrail` sends concurrent HTTP requests to the local vLLM-Metal server, and vLLM-Metal schedules them according to server settings such as `--max-num-seqs`.

```python
from tokenrail import BatchExecutor, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries

client = RailClient.vllm_server(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

queries = {
    "q1": [{"role": "user", "content": "Output only this lowercase token: ok"}],
    "q2": [{"role": "user", "content": "Output only this lowercase token: blue"}],
    "q3": [{"role": "user", "content": "Output only this digit: 7"}],
}

items = batch_items_from_queries(
    queries,
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    max_output_tokens=16,
    temperature=0.0,
)

executor = BatchExecutor(
    client=client,
    max_workers=8,
    sinks=[
        ResultsJsonlSink("out/results.jsonl"),
        PerRequestJsonSink("out/requests"),
    ],
    monitor=RollingMetricsMonitor(),
)

stats = executor.run(items)
print(stats.to_dict())
```

For Apple Silicon server mode, tune both sides together: `max_workers` controls how many requests tokenrail sends concurrently, while vLLM-Metal's `--max-num-seqs` controls how many active sequences the local server schedules at once.

`gpu_memory_utilization` is the CUDA vLLM memory knob for in-process Linux vLLM. On Apple Silicon, configure memory on the vLLM-Metal server side.

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
- `vllm_server`: connects to an OpenAI-compatible `/v1/chat/completions` server such as vLLM-Metal

## Notes

- Apple Silicon GPU support depends on vLLM-Metal: see the vLLM [Apple Silicon GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [CPU Apple Silicon notes](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/), [vLLM-Metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/), and [supported models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/).
- OpenAI cost allocation is inferred from `billing.payer` in the response body. When `payer == "openai"`, the nominal request cost is counted as OpenAI-covered rather than developer-billed.
- v1 local vLLM support is text-only. Multimodal local inputs can be added behind the same provider interface later.
- `reasoning_effort` is gated to `gpt-5` / `o`-series style models in the checked-in capability registry.
- `reasoning_effort`, `verbosity`, `response_format`, `service_tier`, and `store` are intentionally rejected for local vLLM providers.
