from __future__ import annotations

import time
from typing import Any

from ..types import JsonDict, NormalizedResponse, TimingBreakdown, UsageBreakdown
from .base import BaseProvider
from .vllm import _normalize_messages


def _serialize_response(response: Any) -> JsonDict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def _extract_output_text(raw: JsonDict) -> str | None:
    choices = raw.get("choices") or []
    if not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message") or {}
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _usage_from_chat_completion(raw: JsonDict) -> UsageBreakdown:
    usage = raw.get("usage") or {}
    return UsageBreakdown(
        input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        cached_tokens=0,
        output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        reasoning_tokens=0,
        total_tokens=int(usage.get("total_tokens") or 0),
    ).finalized()


class VLLMServerProvider(BaseProvider):
    name = "vllm_server"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: float | None = None,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = client or self._build_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._chat_completions = self._client.chat.completions

    def _build_client(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float | None,
        max_retries: int,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for RailClient.vllm_server(). Install it with `uv add openai`.") from exc
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def build_payload(
        self,
        *,
        model: str,
        input: Any,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: JsonDict | None = None,
        extra_body: JsonDict | None = None,
        **extra: Any,
    ) -> JsonDict:
        unsupported = [
            name
            for name in ("reasoning_effort", "verbosity", "service_tier", "store")
            if extra.pop(name, None) is not None
        ]
        if unsupported:
            raise ValueError(f"{', '.join(unsupported)} is not supported for VLLMServerProvider")

        enable_thinking = extra.pop("enable_thinking", None)
        top_k = extra.pop("top_k", None)
        body = dict(extra_body or {})
        if enable_thinking is not None:
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(enable_thinking)
        if top_k is not None:
            body["top_k"] = int(top_k)

        payload: JsonDict = {
            "model": model,
            "messages": _normalize_messages(input),
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if response_format is not None:
            payload["response_format"] = response_format
        if body:
            payload["extra_body"] = body
        payload.update(extra)
        return payload

    def create(
        self,
        *,
        model: str,
        input: Any,
        request_id: str | None = None,
        metadata: JsonDict | None = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        payload = self.build_payload(model=model, input=input, **kwargs)
        started_at = time.time()
        response = self._chat_completions.create(**payload)
        completed_at = time.time()
        raw = _serialize_response(response)
        return NormalizedResponse(
            id=request_id or str(raw.get("id") or ""),
            model=str(raw.get("model") or model),
            provider=self.name,
            output_text=_extract_output_text(raw),
            raw_response=raw,
            usage=_usage_from_chat_completion(raw),
            billing=None,
            cost=None,
            timing=TimingBreakdown(
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=completed_at - started_at,
            ),
            metadata=metadata,
        )
