from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .catalog import ModelPricing
from .providers.base import BaseProvider
from .providers.openai import OpenAIProvider
from .types import NormalizedResponse


class _ResponsesNamespace:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def create(self, **kwargs: Any) -> NormalizedResponse:
        return self._provider.create(**kwargs)

    def parse(self, **kwargs: Any) -> NormalizedResponse:
        return self._provider.parse(**kwargs)


class _DecisionsNamespace:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def create(self, **kwargs: Any) -> NormalizedResponse:
        return self._provider.create_decision(**kwargs)


class RailClient:
    """Provider-agnostic client with SDK-style responses and decisions namespaces.

    Wraps a :class:`~tokenrail.providers.base.BaseProvider` and exposes it through
    namespaces that mirror the OpenAI SDK call shape while
    returning :class:`~tokenrail.types.NormalizedResponse` objects.
    """

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider
        self.responses = _ResponsesNamespace(provider)
        self.decisions = _DecisionsNamespace(provider)

    @classmethod
    def openai(
        cls,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        client: Any | None = None,
        pricing: Mapping[str, Mapping[str, ModelPricing]] | None = None,
    ) -> RailClient:
        """Build a :class:`RailClient` backed by the OpenAI Python SDK.

        ``max_retries`` configures the SDK's built-in retry behavior; tokenrail
        does not add its own retry loop. Pass ``client`` to inject a pre-built
        (or fake) OpenAI client instead of constructing one.
        """
        provider = OpenAIProvider(
            client=client,
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            base_url=base_url,
            max_retries=max_retries,
            pricing=pricing,
        )
        return cls(provider=provider)
