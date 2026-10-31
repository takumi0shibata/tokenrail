"""Per-client pricing overrides and traceable catalog estimates."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping
from dataclasses import fields, replace
from decimal import Decimal, InvalidOperation

from .catalog import (
    ModelCatalogFallbackWarning,
    ModelPricing,
    _cost_from_pricing,
    _infer_pricing_model,
    _is_registered_model_name,
    get_model_pricing,
)
from .types import CostBreakdown, UsageBreakdown

_RATE_FIELDS = tuple(field.name for field in fields(ModelPricing) if field.name != "service_tier")


def _validated_price(price: ModelPricing) -> ModelPricing:
    if not isinstance(price, ModelPricing):
        raise TypeError("pricing values must be ModelPricing instances")
    rates = {}
    for name in _RATE_FIELDS:
        value = getattr(price, name)
        if value is None and name in {"cached_input_per_million", "cache_write_input_per_million"}:
            rates[name] = None
            continue
        try:
            rate = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ValueError(f"{name} must be a finite non-negative rate") from None
        if not rate.is_finite() or rate < 0:
            raise ValueError(f"{name} must be a finite non-negative rate")
        rates[name] = rate
    return replace(price, **rates)


class _PricingResolver:
    def __init__(self, pricing: Mapping[str, Mapping[str, ModelPricing]] | None) -> None:
        self._overrides: dict[str, dict[str, ModelPricing]] = {}
        if pricing is not None:
            if not isinstance(pricing, Mapping):
                raise TypeError("pricing must map deployment names to API prices")
            for model, apis in pricing.items():
                if not isinstance(model, str) or not model or not isinstance(apis, Mapping):
                    raise ValueError("pricing requires a non-empty deployment name and an API mapping")
                if apis.keys() - {"responses", "decisions"}:
                    raise ValueError("pricing API keys must be 'responses' or 'decisions'")
                self._overrides[model] = {api: _validated_price(price) for api, price in apis.items()}
        self._warned: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def _warn(self, model: str, api: str, reason: str) -> None:
        family = api.split(".")[0]
        with self._lock:
            key = (model, api)
            if key in self._warned:
                return
            self._warned.add(key)
        warnings.warn(
            f"api={api}\n  deployment={model!r}; {reason}\n"
            "  Set RailClient.openai(client=client,\n"
            f"      pricing={{{model!r}: {{{family!r}: your_model_pricing}}}})\n"
            "  using ModelPricing(...) with your USD-per-1M-token rates.\n"
            "  See README: Custom pricing.",
            ModelCatalogFallbackWarning,
            stacklevel=4,
        )

    def calculate(
        self, *, model: str, api: str, usage: UsageBreakdown, payer: str | None,
        custom_endpoint: bool, service_tier: str,
    ) -> CostBreakdown | None:
        family = api.split(".")[0]
        price = self._overrides.get(model, {}).get(family)
        explicit = price is not None
        matched_model = None
        if explicit:
            if ((usage.cached_tokens and price.cached_input_per_million is None)
                    or (usage.cache_write_tokens and price.cache_write_input_per_million is None)):
                self._warn(model, api, "required cache pricing is missing. Cost is unknown, not free.")
                return None
        else:
            match = _infer_pricing_model(model)
            if match is None:
                self._warn(model, api, "no unique matching model price. Cost is unknown, not free.")
                return None
            matched_model, response_price = match
            price = (get_model_pricing(matched_model, api=api) if family == "decisions" else response_price)
            if price is None:
                self._warn(model, api, f"matched model={matched_model!r}, but no {family} price is registered. "
                           "Cost is unknown, not free.")
                return None
            if custom_endpoint or not _is_registered_model_name(model, match):
                self._warn(
                    model, api, f"matched model={matched_model!r}.\n"
                    "  Using OpenAI catalog rates as an estimate.\n"
                    "  This does not guarantee your Azure or custom-endpoint billed amount.",
                )
        cost = _cost_from_pricing(price, usage, payer)
        # Include the applied fallback rates, not just the optional catalog entries.
        rates = {name: str(getattr(price, name)) if getattr(price, name) is not None else None
                 for name in _RATE_FIELDS}
        if not explicit:
            rates["cached_input_per_million"] = rates["cached_input_per_million"] or "0"
            rates["cache_write_input_per_million"] = (
                rates["cache_write_input_per_million"] or rates["input_per_million"]
            )
        cost.pricing = {
            "source": "explicit" if explicit else "openai_catalog",
            "requested_model": model,
            "matched_model": matched_model,
            "api": api,
            "custom_endpoint": custom_endpoint,
            "service_tier": service_tier,
            "unit": "USD per 1M tokens",
            "rates": rates,
        }
        return cost
