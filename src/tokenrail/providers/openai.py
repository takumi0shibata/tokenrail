from __future__ import annotations

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


def _extract_output_parsed(response: Any, raw: JsonDict) -> Any | None:
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed

    raw_output_parsed = raw.get("output_parsed")
    if raw_output_parsed is not None:
        return raw_output_parsed

    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed

    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            parsed = content.get("parsed")
            if parsed is not None:
                return parsed
    return None


def _extract_refusal(response: Any, raw: JsonDict) -> str | None:
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if isinstance(refusal, str):
                return refusal

    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            refusal = content.get("refusal")
            if isinstance(refusal, str):
                return refusal
    return None


def _serialize_response(response: Any) -> JsonDict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    raise TypeError(f"Unsupported response type: {type(response)!r}")


class OpenAIProvider(BaseProvider):
    """Executes requests against the OpenAI Responses API.

    Validates request parameters against the model capability registry,
    normalizes the response (output text, usage, billing), and attaches a cost
    estimate from the checked-in pricing table.
    """

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
    ) -> None:
        self._client = client or self._build_client(
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            base_url=base_url,
            max_retries=max_retries,
        )
        self._responses = self._client.responses

    def _build_client(
        self,
        *,
        api_key: str | None,
        organization: str | None,
        timeout: float | None,
        base_url: str | None,
        max_retries: int,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for RailClient.openai(). Install it with `uv add openai`.") from exc
        return OpenAI(
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            base_url=base_url,
            max_retries=max_retries,
        )

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
        if "text_format" in extra:
            raise ValueError("text_format requires responses.parse(); use client.responses.parse(...)")
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

    def build_parse_payload(
        self,
        *,
        model: str,
        input: Any,
        text_format: Any,
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
        if text_format is None:
            raise ValueError("text_format is required for responses.parse()")
        if response_format is not None:
            raise ValueError("response_format and text_format cannot be used together")
        self._validate_capabilities(
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            response_format={"type": "text_format"},
        )

        payload: JsonDict = {"model": model, "input": input, "text_format": text_format}
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if verbosity is not None:
            payload["verbosity"] = verbosity
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

    def _normalize_response(
        self,
        *,
        response: Any,
        model: str,
        request_id: str | None,
        service_tier: str | None,
        metadata: JsonDict | None,
        started_at: float,
        completed_at: float,
        output_parsed: Any | None = None,
        refusal: str | None = None,
    ) -> NormalizedResponse:
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
            output_parsed=output_parsed,
            refusal=refusal,
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
        response = self._responses.create(**payload)
        completed_at = time.time()
        return self._normalize_response(
            response=response,
            model=model,
            request_id=request_id,
            service_tier=service_tier,
            metadata=metadata,
            started_at=started_at,
            completed_at=completed_at,
        )

    def parse(
        self,
        *,
        model: str,
        input: Any,
        text_format: Any,
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
        payload = self.build_parse_payload(
            model=model,
            input=input,
            text_format=text_format,
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
        response = self._responses.parse(**payload)
        completed_at = time.time()
        raw = _serialize_response(response)
        return self._normalize_response(
            response=raw,
            model=model,
            request_id=request_id,
            service_tier=service_tier,
            metadata=metadata,
            started_at=started_at,
            completed_at=completed_at,
            output_parsed=_extract_output_parsed(response, raw),
            refusal=_extract_refusal(response, raw),
        )
