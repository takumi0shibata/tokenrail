from __future__ import annotations

import math
import time
from collections.abc import Iterable
from typing import Any

from .._requests import prepare_decision_request, request_api
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


def _valid_decision_answer(answer: Any, question: JsonDict) -> bool:
    """Check the answer contract without discarding usage or the original response."""
    if not isinstance(answer, dict) or answer.get("name") != question.get("name"):
        return False
    kind = answer.get("type")
    if kind == "refusal":
        return True
    if kind != question.get("type"):
        return False

    def finite_number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    def probability(value: Any) -> bool:
        return finite_number(value) and 0 <= value <= 1

    if kind == "predicate":
        return probability(answer.get("probability"))
    if kind not in ("choice", "score") or not probability(answer.get("confidence")):
        return False

    def choice_value(value: Any) -> bool:
        return type(value) in (str, bool)

    if kind == "choice" and not choice_value(answer.get("choice")):
        return False
    if kind == "score" and not finite_number(answer.get("score")):
        return False
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        return False
    for item in probabilities:
        if not isinstance(item, dict) or not probability(item.get("probability")):
            return False
        if kind == "choice":
            if not choice_value(item.get("value")):
                return False
        else:
            value = item.get("value")
            if type(value) is not int or value < 0:
                return False
            if not isinstance(item.get("label"), str):
                return False
    return True


def _prepare_payload(payload: JsonDict, prompt_cache_options: JsonDict | None) -> JsonDict:
    """Reject conflicting body overrides and forward cache options via the SDK."""
    if prompt_cache_options is not None:
        payload["prompt_cache_options"] = prompt_cache_options

    extra_body = payload.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a dict")
        if "text_format" in payload and "text" in extra_body:
            raise ValueError("text_format conflicts with extra_body.text")
        for field, value in extra_body.items():
            if field in payload and value != payload[field]:
                raise ValueError(f"{field} conflicts with extra_body.{field}")

    # Cache options are absent from some supported SDK signatures. Forward
    # both cache controls through the body consistently across SDK versions.
    cache_body = {
        field: payload.pop(field)
        for field in ("prompt_cache_options", "prompt_cache_retention")
        if field in payload
    }
    if cache_body:
        payload["extra_body"] = {**(extra_body or {}), **cache_body}
    if payload.get("background") or (extra_body or {}).get("background"):
        raise ValueError("background=True is not supported; tokenrail requires foreground responses")
    if payload.get("stream") or (extra_body or {}).get("stream"):
        raise ValueError("stream=True is not supported; tokenrail requires non-streaming responses")
    return payload


class OpenAIProvider(BaseProvider):
    """Executes requests against the OpenAI Responses and Decisions APIs.

    Validates request parameters against the model capability registry,
    normalizes the response (output text, usage, billing), and attaches a cost
    estimate from the checked-in pricing table. Prompt-cache settings and input
    breakpoints are forwarded as supplied; caching decisions belong to the API
    and the caller.
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
        options: JsonDict = dict(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            max_retries=max_retries,
        )
        if timeout is not None:
            options["timeout"] = timeout
        return OpenAI(**options)

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
        if (
            reasoning_effort is not None
            and capabilities.supported_reasoning_efforts is not None
            and reasoning_effort not in capabilities.supported_reasoning_efforts
        ):
            raise ValueError(f"reasoning_effort={reasoning_effort!r} is not supported for model {model}")
        if capabilities.sampling_requires_no_reasoning and reasoning_effort != "none":
            for name, value in (("temperature", temperature), ("top_p", top_p)):
                if value is not None:
                    raise ValueError(f"{name} requires reasoning_effort='none' for model {model}")

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
        prompt_cache_key: str | None = None,
        prompt_cache_options: JsonDict | None = None,
        prompt_cache_retention: str | None = None,
        **extra: Any,
    ) -> JsonDict:
        if "text_format" in extra:
            raise ValueError("text_format requires responses.parse(); use client.responses.parse(...)")
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
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            payload["prompt_cache_retention"] = prompt_cache_retention
        return self._prepare_and_validate_payload(payload, extra, prompt_cache_options)

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
        prompt_cache_key: str | None = None,
        prompt_cache_options: JsonDict | None = None,
        prompt_cache_retention: str | None = None,
        **extra: Any,
    ) -> JsonDict:
        if text_format is None:
            raise ValueError("text_format is required for responses.parse()")
        if response_format is not None:
            raise ValueError("response_format and text_format cannot be used together")
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
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            payload["prompt_cache_retention"] = prompt_cache_retention
        return self._prepare_and_validate_payload(payload, extra, prompt_cache_options)

    def _prepare_and_validate_payload(
        self, payload: JsonDict, extra: JsonDict, prompt_cache_options: JsonDict | None
    ) -> JsonDict:
        # Native SDK settings may extend aliases, but must not silently replace them.
        for field, value in extra.items():
            if field in payload:
                existing = payload[field]
                if isinstance(existing, dict) and isinstance(value, dict):
                    for key in existing.keys() & value.keys():
                        if existing[key] != value[key]:
                            raise ValueError(f"{field}.{key} conflicts with the corresponding convenience argument")
                    value = {**existing, **value}
                elif existing != value:
                    raise ValueError(f"{field} conflicts with the corresponding convenience argument")
            payload[field] = value

        payload = _prepare_payload(payload, prompt_cache_options)
        if request_api(payload) == "decisions.create":
            raise ValueError("questions requires decisions.create(); use client.decisions.create(...)")
        # Validate what the SDK will send, including fields supplied only in extra_body.
        body = {**payload, **(payload.get("extra_body") or {})}
        reasoning = body.get("reasoning")
        text = body.get("text")
        self._validate_capabilities(
            model=body["model"],
            reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None,
            verbosity=text.get("verbosity") if isinstance(text, dict) else body.get("verbosity"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_output_tokens=body.get("max_output_tokens"),
            response_format=(
                {"type": "text_format"} if "text_format" in payload
                else text.get("format") if isinstance(text, dict) else None
            ),
        )
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
        latency_seconds: float,
        output_parsed: Any | None = None,
        refusal: str | None = None,
        api: str = "responses.create",
    ) -> NormalizedResponse:
        raw = _serialize_response(response)
        usage = UsageBreakdown.from_dict(raw.get("usage")).finalized()
        billing = raw.get("billing")
        actual_tier = str(raw.get("service_tier") or service_tier or "default")
        payer = billing.get("payer") if isinstance(billing, dict) else None
        cost = calculate_cost(model=model, usage=usage, payer=payer, service_tier=actual_tier, api=api)
        if api == "decisions.create":
            # Azure deployments and other custom endpoints have independent pricing.
            endpoint = str(getattr(self._client, "base_url", "https://api.openai.com/v1/")).rstrip("/")
            if endpoint != "https://api.openai.com/v1":
                cost = None
        status = raw.get("status")
        error = None
        if raw.get("error") is not None:
            error = f"Response error: {raw['error']}"
        elif status is not None and status != "completed":
            error = f"Response status={status}: {raw.get('incomplete_details') or 'response did not complete'}"
        return NormalizedResponse(
            id=request_id if request_id is not None else str(raw.get("id") or ""),
            model=str(raw.get("model") or model),
            provider=self.name,
            output_text=_extract_output_text(raw),
            raw_response=raw,
            output_parsed=output_parsed,
            refusal=refusal if refusal is not None else _extract_refusal(response, raw),
            usage=usage,
            billing=billing if isinstance(billing, dict) else None,
            cost=cost,
            timing=TimingBreakdown(
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=latency_seconds,
            ),
            metadata=metadata,
            status=status,
            error=error,
            api=api,
            answers=raw.get("answers") if api == "decisions.create" else None,
        )

    def create_decision(
        self,
        *,
        model: str,
        input: Any,
        questions: Iterable[JsonDict],
        request_id: str | None = None,
        metadata: JsonDict | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: JsonDict | None = None,
        **extra: Any,
    ) -> NormalizedResponse:
        """Evaluate questions, preserving typed answers and request-level usage.

        ``metadata`` is local result metadata; it is not sent to Decisions.
        Cache controls use extra_body to support the minimum SDK version.
        """
        payload = prepare_decision_request({"model": model, "input": input, "questions": questions, **extra})
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key
        payload = _prepare_payload(payload, prompt_cache_options)
        if "prompt_cache_key" in payload:
            payload["extra_body"] = {
                **(payload.get("extra_body") or {}), "prompt_cache_key": payload.pop("prompt_cache_key")
            }

        started_at = time.time()
        started_monotonic = time.monotonic()
        response = self._client.decisions.create(**payload)
        normalized = self._normalize_response(
            response=response,
            model=model,
            request_id=request_id,
            service_tier=None,
            metadata=metadata,
            started_at=started_at,
            completed_at=time.time(),
            latency_seconds=time.monotonic() - started_monotonic,
            api="decisions.create",
        )
        if normalized.error is None:
            if not isinstance(normalized.answers, list) or len(normalized.answers) != len(payload["questions"]):
                normalized.error = "Decision response must contain one answer per question"
            else:
                for index, (answer, question) in enumerate(zip(normalized.answers, payload["questions"], strict=True)):
                    if not _valid_decision_answer(answer, question):
                        normalized.error = f"Decision answer at index {index} is invalid or does not match its question"
                        break
                else:
                    normalized.status = "completed"
        return normalized

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
        prompt_cache_key: str | None = None,
        prompt_cache_options: JsonDict | None = None,
        prompt_cache_retention: str | None = None,
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            prompt_cache_retention=prompt_cache_retention,
            **extra,
        )

        started_at = time.time()
        started_monotonic = time.monotonic()
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
            latency_seconds=time.monotonic() - started_monotonic,
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
        prompt_cache_key: str | None = None,
        prompt_cache_options: JsonDict | None = None,
        prompt_cache_retention: str | None = None,
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            prompt_cache_retention=prompt_cache_retention,
            **extra,
        )

        started_at = time.time()
        started_monotonic = time.monotonic()
        raw_api = getattr(self._responses, "with_raw_response", None)
        if raw_api is None:
            # Keep supporting injected clients implementing only create/parse.
            response = self._responses.parse(**payload)
            raw = _serialize_response(response)
        else:
            # Obtain the body before the SDK's structured-output post-parser
            # can fail. Parsing this same response never issues another request.
            api_response = raw_api.parse(**payload)
            raw = api_response.http_response.json()
            try:
                response = api_response.parse()
            except Exception as exc:
                normalized = self._normalize_response(
                    response=raw,
                    model=model,
                    request_id=request_id,
                    service_tier=service_tier,
                    metadata=metadata,
                    started_at=started_at,
                    completed_at=time.time(),
                    latency_seconds=time.monotonic() - started_monotonic,
                    api="responses.parse",
                )
                parse_error = f"Structured output parsing failed ({type(exc).__name__}): {exc}"
                normalized.error = f"{normalized.error}; {parse_error}" if normalized.error else parse_error
                return normalized
        completed_at = time.time()
        return self._normalize_response(
            response=raw,
            model=model,
            request_id=request_id,
            service_tier=service_tier,
            metadata=metadata,
            started_at=started_at,
            completed_at=completed_at,
            latency_seconds=time.monotonic() - started_monotonic,
            output_parsed=_extract_output_parsed(response, raw),
            refusal=_extract_refusal(response, raw),
            api="responses.parse",
        )
