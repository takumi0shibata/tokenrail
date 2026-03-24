from __future__ import annotations

import random
import time
from typing import Any

from ..catalog import calculate_cost, get_model_capabilities
from ..types import JsonDict, NormalizedResponse, TimingBreakdown, UsageBreakdown
from .base import BaseProvider


def _extract_output_text(raw: JsonDict) -> str | None:
    output_text = raw.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks) if chunks else None


def _serialize_response(response: Any) -> JsonDict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    raise TypeError(f"Unsupported response type: {type(response)!r}")


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        max_retries: int = 6,
        base_sleep: float = 1.0,
        retry_exceptions: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self._client = client or self._build_client(
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            base_url=base_url,
        )
        self._responses = self._client.responses
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.retry_exceptions = retry_exceptions if retry_exceptions is not None else self._default_retry_exceptions()

    def _build_client(
        self,
        *,
        api_key: str | None,
        organization: str | None,
        timeout: float | None,
        base_url: str | None,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for RailClient.openai(). Install it with `uv add openai`.") from exc
        return OpenAI(api_key=api_key, organization=organization, timeout=timeout, base_url=base_url)

    def _default_retry_exceptions(self) -> tuple[type[BaseException], ...]:
        try:
            from openai import APIError, APITimeoutError, InternalServerError, RateLimitError
        except ImportError:
            return ()
        return (RateLimitError, APITimeoutError, InternalServerError, APIError)

    def _validate_capabilities(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        verbosity: str | None,
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int | None,
        response_format: JsonDict | None,
    ) -> None:
        capabilities = get_model_capabilities(model)
        checks = [
            ("reasoning_effort", reasoning_effort, capabilities.reasoning_effort),
            ("verbosity", verbosity, capabilities.verbosity),
            ("temperature", temperature, capabilities.temperature),
            ("top_p", top_p, capabilities.top_p),
            ("max_output_tokens", max_output_tokens, capabilities.max_output_tokens),
            ("response_format", response_format, capabilities.response_format),
        ]
        for name, value, supported in checks:
            if value is not None and not supported:
                raise ValueError(f"{name} is not supported for model {model}")

    def build_payload(
        self,
        *,
        model: str,
        input: Any,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: JsonDict | None = None,
        metadata: JsonDict | None = None,
        service_tier: str | None = None,
        store: bool | None = None,
        **extra: Any,
    ) -> JsonDict:
        self._validate_capabilities(
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )

        payload: JsonDict = {"model": model, "input": input}
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        text_config: JsonDict = {}
        if verbosity is not None:
            text_config["verbosity"] = verbosity
        if response_format is not None:
            text_config["format"] = response_format
        if text_config:
            payload["text"] = text_config
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if metadata is not None:
            payload["metadata"] = metadata
        if service_tier is not None:
            payload["service_tier"] = service_tier
        if store is not None:
            payload["store"] = store
        payload.update(extra)
        return payload

    def create(
        self,
        *,
        model: str,
        input: Any,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: JsonDict | None = None,
        request_id: str | None = None,
        metadata: JsonDict | None = None,
        service_tier: str | None = None,
        store: bool | None = None,
        **extra: Any,
    ) -> NormalizedResponse:
        payload = self.build_payload(
            model=model,
            input=input,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            metadata=metadata,
            service_tier=service_tier,
            store=store,
            **extra,
        )

        started_at = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._responses.create(**payload)
                completed_at = time.time()
                raw = _serialize_response(response)
                usage = UsageBreakdown.from_dict(raw.get("usage")).finalized()
                billing = raw.get("billing")
                actual_tier = str(raw.get("service_tier") or service_tier or "default")
                cost = calculate_cost(model=model, usage=usage, payer=(billing or {}).get("payer"), service_tier=actual_tier)
                return NormalizedResponse(
                    id=request_id or str(raw.get("id") or ""),
                    model=str(raw.get("model") or model),
                    provider=self.name,
                    output_text=_extract_output_text(raw),
                    raw_response=raw,
                    usage=usage,
                    billing=billing if isinstance(billing, dict) else None,
                    cost=cost,
                    timing=TimingBreakdown(
                        started_at=started_at,
                        completed_at=completed_at,
                        latency_seconds=completed_at - started_at,
                    ),
                    metadata=metadata,
                )
            except self.retry_exceptions:
                if attempt == self.max_retries:
                    raise
                sleep = self.base_sleep * (2 ** (attempt - 1))
                time.sleep(sleep * (0.5 + random.random()))

        raise RuntimeError("OpenAIProvider retry loop exited unexpectedly")
