from __future__ import annotations

from typing import Any

from .providers.base import BaseProvider
from .providers.hf import HFTransformersProvider
from .providers.openai import OpenAIProvider
from .types import NormalizedResponse


class _ResponsesNamespace:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def create(self, **kwargs: Any) -> NormalizedResponse:
        return self._provider.create(**kwargs)


class RailClient:
    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider
        self.responses = _ResponsesNamespace(provider)

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
    ) -> "RailClient":
        provider = OpenAIProvider(
            client=client,
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            base_url=base_url,
            max_retries=max_retries,
        )
        return cls(provider=provider)

    @classmethod
    def hf(
        cls,
        *,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
    ) -> "RailClient":
        provider = HFTransformersProvider(
            model_id=model_id,
            device_map=device_map,
            dtype=dtype,
            batch_size=batch_size,
        )
        return cls(provider=provider)
