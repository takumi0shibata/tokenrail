# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
