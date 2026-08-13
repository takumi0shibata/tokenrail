# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Opt-in GPT-5.6 explicit prompt-cache planning in `BatchExecutor`, including
  longest-common-prefix detection at content-block boundaries.
- Deterministic `prompt_cache_key` sharding with automatic shard counts and a
  rolling per-key submit limit (15 RPM by default).
- Public `PromptCacheConfig`, `cache_write_tokens` usage metrics, and prompt
  cache configuration fields on `StatsSnapshot`.

### Changed

- GPT-5.6 cache writes are included in cost estimates at 1.25 times the normal
  input rate.
- GPT-5.6 Terra and Luna prices now match the current OpenAI base pricing.
- Progress output reports cache reads, cache writes, and active shard counts.

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
