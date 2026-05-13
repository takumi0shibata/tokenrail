from __future__ import annotations

from typing import Any

from .providers.base import BaseProvider
from .providers.openai import OpenAIProvider
from .providers.vllm import VLLMProvider
from .providers.vllm_server import VLLMServerProvider
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
    def vllm(
        cls,
        *,
        model_id: str,
        family: str,
        batch_flush_size: int = 256,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        quantization: str | None = None,
        trust_remote_code: bool = False,
        enable_prefix_caching: bool = False,
        seed: int | None = None,
        device: str | None = None,
        metal_memory_fraction: str | float | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
    ) -> "RailClient":
        provider = VLLMProvider(
            model_id=model_id,
            family=family,
            batch_flush_size=batch_flush_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            device=device,
            metal_memory_fraction=metal_memory_fraction,
            extra_llm_kwargs=extra_llm_kwargs,
        )
        return cls(provider=provider)

    @classmethod
    def vllm_server(
        cls,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: float | None = None,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> "RailClient":
        provider = VLLMServerProvider(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )
        return cls(provider=provider)
