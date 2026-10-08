# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for its 0.x development releases, with a one-time version reset in 0.4.0.
Earlier 1.x and 2.x entries are retained as published history; their stable
designation was premature and does not describe the project's current status.

## [Unreleased]

### Added

- GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol, and GPT-6 Luna catalog entries with
  standard-tier base prices and cache-write rates.
- GPT-6 reasoning-effort and sampling-parameter validation for both
  `responses.create(...)` and `responses.parse(...)`.
- OpenAI prompt-cache settings are forwarded unchanged for both
  `responses.create(...)` and `responses.parse(...)`.
- Conflicting request fields in `extra_body` are rejected before sending.
- `cache_write_tokens` usage metrics on responses and batch/model statistics.

### Changed

- Reset the package version from 2.0.0 to 0.4.0 and the development status from
  Production/Stable to Beta. This continues development from the existing code;
  it does not restore an older implementation. Public API compatibility is not
  yet guaranteed between 0.x minor releases.
- GPT-5.6 Sol and the `gpt-5.6` alias use OpenAI's current promotional prices:
  $4.00 input, $0.40 cached input, $5.00 cache writes, and $20.00 output per
  million tokens, verified on October 8, 2026.
- GPT-5.6 cache writes are included in cost estimates at 1.25 times the normal
  input rate.
- GPT-5.6 Terra and Luna prices now match OpenAI's August 2026 price cuts,
  including cached-input and 1.25-times cache-write rates.
- Prompt caching uses only OpenAI's standard request settings; tokenrail does
  not plan prefixes, generate or shard keys, prewarm caches, or impose per-key
  submit limits. `BatchExecutor.prompt_cache` and `PromptCacheConfig` are removed.
- Progress output reports cache reads and writes independently of request settings.
- Progress output preserves unknown payer strings instead of displaying `?`,
  and falls back to `billing.payer` when the cost breakdown has no payer.

## [2.0.0] - 2026-08-03

### Added

- GPT-5.6 Sol, Terra, and Luna capability and base-price catalog entries,
  including the `gpt-5.6` alias for Sol.
- `ModelCatalogFallbackWarning` identifies unregistered models and the catalog
  entries used as capability or pricing fallbacks.
- Payer state, payer switch count, and raw payer request counts on
  `StatsSnapshot`.

### Changed

- `RollingMetricsMonitor` now prints a compact request line, explicit payer
  transitions, periodic summaries, and a final batch summary by default.
  Pass `verbose=True` to retain the previous progress format.
- Monitor output supports automatic or forced ANSI emphasis without an added
  dependency.
- GPT-5.6 cost estimates use current base standard-tier prices. Long-context,
  cache-write, Batch/Fast/Flex, and regional price adjustments remain outside
  the estimator.

## [1.1.0] - 2026-06-14

### Added

- Structured output parsing for batch jobs. `BatchExecutor` now calls
  `responses.parse(...)` automatically for items that include `text_format`.
- `client.responses.parse(...)` and `OpenAIProvider.parse(...)`.
- `NormalizedResponse.output_parsed` and `NormalizedResponse.refusal`.
- Default JSONL result records now include `output_parsed` and `refusal`.
- README example for Pydantic structured output batches.

### Changed

- `response_format` remains the low-level JSON Schema path for
  `responses.create(...)`; `text_format` is the high-level Pydantic parsing path.
  They are rejected when used together.

## [1.0.0] - 2026-06-11

First stable release. The public API surface is now covered by semantic
versioning: `RailClient`, `BatchExecutor`, `batch_items_from_queries`,
`RollingMetricsMonitor`, `ResultsJsonlSink`, `PerRequestJsonSink`,
`OpenAIProvider`, and the types exported from `tokenrail`.

### Added

- `py.typed` marker — the package now ships inline type information (PEP 561).
- `tokenrail.__version__`.
- Docstrings across the public API.
- Complete packaging metadata (license, classifiers, project URLs).
- CI test workflow across Python 3.10–3.14.
- First release published to [PyPI](https://pypi.org/project/tokenrail/);
  releases are published automatically from `v*` tags via PyPI Trusted
  Publishing.

### Changed

- Minimum supported Python lowered from 3.11 to 3.10.
- `catalog.get_model_pricing` no longer carries a dead `service_tier` branch;
  non-default tiers explicitly fall back to default-tier pricing.

## [0.2.1] - 2026-06-10

### Added

- Client-side RPM and TPM submit throttling in `BatchExecutor`
  (`max_rpm` / `max_tpm`).

### Removed

- vLLM provider support; the library is OpenAI-only.

## [0.1.3] - 2026-05-20

### Changed

- Removed the custom OpenAI retry loop in favor of the SDK's built-in
  `max_retries`.

## [0.1.2] - 2026-05-15

### Added

- ETA progress reporting in `RollingMetricsMonitor`.

## [0.1.0] - 2026-05-13

### Added

- Initial release: `RailClient`, `BatchExecutor`, rolling metrics monitor,
  JSONL and per-request sinks, model capability/pricing catalog.
