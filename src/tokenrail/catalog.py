from __future__ import annotations

import re
import threading
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeVar

from .types import CostBreakdown, UsageBreakdown

_T = TypeVar("_T")


class ModelCatalogFallbackWarning(UserWarning):
    """Warning emitted when an unregistered model uses fallback catalog data."""


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    reasoning_effort: bool
    verbosity: bool
    temperature: bool
    top_p: bool
    max_output_tokens: bool
    response_format: bool
    prompt_cache_explicit: bool = False


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_million: Decimal
    cached_input_per_million: Decimal | None
    output_per_million: Decimal
    service_tier: str = "default"
    cache_write_input_per_million: Decimal | None = None


_CAPABILITY_RULES: list[tuple[tuple[str, ...], ModelCapabilities]] = [
    (
        ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"),
        ModelCapabilities(
            reasoning_effort=True,
            verbosity=True,
            temperature=True,
            top_p=True,
            max_output_tokens=True,
            response_format=True,
            prompt_cache_explicit=True,
        ),
    ),
    (
        ("gpt-5", "o1", "o3", "o4"),
        ModelCapabilities(
            reasoning_effort=True,
            verbosity=True,
            temperature=True,
            top_p=True,
            max_output_tokens=True,
            response_format=True,
        ),
    ),
    (
        ("gpt-4.1", "gpt-4o"),
        ModelCapabilities(
            reasoning_effort=False,
            verbosity=True,
            temperature=True,
            top_p=True,
            max_output_tokens=True,
            response_format=True,
        ),
    ),
]

_DEFAULT_CAPABILITIES = ModelCapabilities(
    reasoning_effort=False,
    verbosity=False,
    temperature=True,
    top_p=True,
    max_output_tokens=True,
    response_format=True,
)

_PRICING_RULES: list[tuple[tuple[str, ...], ModelPricing]] = [
    (
        ("gpt-5.6-terra",),
        ModelPricing(
            Decimal("2.50"),
            Decimal("0.25"),
            Decimal("15.00"),
            cache_write_input_per_million=Decimal("3.125"),
        ),
    ),
    (
        ("gpt-5.6-luna",),
        ModelPricing(
            Decimal("1.00"),
            Decimal("0.10"),
            Decimal("6.00"),
            cache_write_input_per_million=Decimal("1.25"),
        ),
    ),
    (
        ("gpt-5.6-sol", "gpt-5.6"),
        ModelPricing(
            Decimal("5.00"),
            Decimal("0.50"),
            Decimal("30.00"),
            cache_write_input_per_million=Decimal("6.25"),
        ),
    ),
    (("gpt-5.5",), ModelPricing(Decimal("5.00"), Decimal("0.50"), Decimal("30.00"))),
    (("gpt-5.4-mini",), ModelPricing(Decimal("0.750"), Decimal("0.075"), Decimal("4.500"))),
    (("gpt-5.4-nano",), ModelPricing(Decimal("0.20"), Decimal("0.02"), Decimal("1.25"))),
    (("gpt-5.4",), ModelPricing(Decimal("2.50"), Decimal("0.25"), Decimal("15.00"))),
    (("gpt-5.2",), ModelPricing(Decimal("1.75"), Decimal("0.175"), Decimal("14.00"))),
    (("gpt-5-mini",), ModelPricing(Decimal("0.25"), Decimal("0.025"), Decimal("2.00"))),
    (("gpt-5-nano",), ModelPricing(Decimal("0.05"), Decimal("0.005"), Decimal("0.40"))),
    (("gpt-5",), ModelPricing(Decimal("1.25"), Decimal("0.125"), Decimal("10.00"))),
    (("gpt-4.1-mini",), ModelPricing(Decimal("0.40"), Decimal("0.10"), Decimal("1.60"))),
    (("gpt-4.1-nano",), ModelPricing(Decimal("0.10"), Decimal("0.025"), Decimal("0.40"))),
    (("gpt-4.1",), ModelPricing(Decimal("2.00"), Decimal("0.50"), Decimal("8.00"))),
    (("gpt-4o-mini",), ModelPricing(Decimal("0.15"), Decimal("0.075"), Decimal("0.60"))),
    (("gpt-4o",), ModelPricing(Decimal("2.50"), Decimal("1.25"), Decimal("10.00"))),
    (("o4-mini",), ModelPricing(Decimal("1.10"), Decimal("0.275"), Decimal("4.40"))),
    (("o3",), ModelPricing(Decimal("2.00"), Decimal("0.50"), Decimal("8.00"))),
    (("o1",), ModelPricing(Decimal("15.00"), Decimal("7.50"), Decimal("60.00"))),
]


_MODEL_NAME_DELIMITERS = {"-", "_", "/", ":", " ", "."}
_OFFICIAL_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}")
_WARNED_FALLBACK_MODELS: set[str] = set()
_WARNING_LOCK = threading.Lock()


def _is_delimited_match(model: str, candidate: str, start: int, end: int) -> bool:
    before = model[start - 1] if start > 0 else None
    after = model[end] if end < len(model) else None
    before_ok = before is None or before in _MODEL_NAME_DELIMITERS
    after_ok = after is None or after in _MODEL_NAME_DELIMITERS
    return before_ok and after_ok


def _match_rule(model: str, rules: Iterable[tuple[tuple[str, ...], _T]]) -> tuple[str, _T] | None:
    matches: list[tuple[int, int, str, _T]] = []
    for rule_index, (prefixes, payload) in enumerate(rules):
        for prefix in prefixes:
            start = model.find(prefix)
            while start != -1:
                end = start + len(prefix)
                if _is_delimited_match(model, prefix, start, end):
                    matches.append((len(prefix), rule_index, prefix, payload))
                    break
                start = model.find(prefix, start + 1)

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], item[1]))
    _, _, prefix, payload = matches[0]
    return prefix, payload


def _is_registered_model_name(model: str, pricing_match: tuple[str, ModelPricing] | None) -> bool:
    if pricing_match is None:
        return False
    prefix, _ = pricing_match
    if model == prefix:
        return True
    return model.startswith(prefix) and _OFFICIAL_SNAPSHOT_SUFFIX.fullmatch(model[len(prefix) :]) is not None


def _warn_for_catalog_fallback(
    model: str,
    capability_match: tuple[str, ModelCapabilities] | None,
    pricing_match: tuple[str, ModelPricing] | None,
) -> None:
    if _is_registered_model_name(model, pricing_match):
        return

    with _WARNING_LOCK:
        if model in _WARNED_FALLBACK_MODELS:
            return
        _WARNED_FALLBACK_MODELS.add(model)

    capability_source = repr(capability_match[0]) if capability_match is not None else "the default capabilities"
    pricing_source = repr(pricing_match[0]) if pricing_match is not None else "no pricing entry (cost=None)"
    warnings.warn(
        f"Model {model!r} is not explicitly registered in tokenrail's model catalog; "
        f"using capabilities from {capability_source} and pricing from {pricing_source}.",
        ModelCatalogFallbackWarning,
        stacklevel=3,
    )


def _resolve_model_catalog(
    model: str,
) -> tuple[tuple[str, ModelCapabilities] | None, tuple[str, ModelPricing] | None]:
    capability_match = _match_rule(model, _CAPABILITY_RULES)
    pricing_match = _match_rule(model, _PRICING_RULES)
    _warn_for_catalog_fallback(model, capability_match, pricing_match)
    return capability_match, pricing_match


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Return the request-parameter capabilities for ``model``.

    Matching is delimiter-aware substring matching against the checked-in
    capability registry; unknown models fall back to a conservative default.
    """
    capability_match, _ = _resolve_model_catalog(model)
    return capability_match[1] if capability_match is not None else _DEFAULT_CAPABILITIES


def get_model_pricing(model: str, service_tier: str = "default") -> ModelPricing | None:
    """Return per-million-token pricing for ``model``, or ``None`` if unknown.

    The checked-in registry only carries default-tier prices; for other service
    tiers the default-tier price is returned as an approximation.
    """
    _, pricing_match = _resolve_model_catalog(model)
    return pricing_match[1] if pricing_match is not None else None


def calculate_cost(
    model: str,
    usage: UsageBreakdown,
    payer: str | None,
    service_tier: str = "default",
) -> CostBreakdown | None:
    """Compute the nominal USD cost of ``usage`` and attribute it to a payer.

    Returns ``None`` when the model has no pricing entry. When ``payer`` is
    ``"openai"`` the cost is attributed to OpenAI instead of the developer.
    """
    pricing = get_model_pricing(model, service_tier=service_tier)
    if pricing is None:
        return None

    ordinary_input = max(usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens, 0)
    cached_rate = pricing.cached_input_per_million or Decimal("0")
    cache_write_rate = pricing.cache_write_input_per_million or pricing.input_per_million
    total = (
        (Decimal(ordinary_input) * pricing.input_per_million)
        + (Decimal(usage.cached_tokens) * cached_rate)
        + (Decimal(usage.cache_write_tokens) * cache_write_rate)
        + (Decimal(usage.output_tokens) * pricing.output_per_million)
    ) / Decimal("1000000")

    nominal = float(total)
    if payer == "openai":
        return CostBreakdown(nominal_usd=nominal, developer_usd=0.0, openai_usd=nominal, payer=payer)
    return CostBreakdown(nominal_usd=nominal, developer_usd=nominal, openai_usd=0.0, payer=payer)
