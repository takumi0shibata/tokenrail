from __future__ import annotations

import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..types import JsonDict, NormalizedResponse, TimingBreakdown, UsageBreakdown
from .base import BaseProvider


def _ensure_text_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("VLLMProvider only supports text content in v1")
            item_type = item.get("type")
            if item_type in {"input_text", "text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
                continue
            raise ValueError("VLLMProvider only supports text content in v1")
        return {"role": message["role"], "content": "\n".join(text_parts)}
    raise ValueError("Unsupported message content")


def _normalize_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, list):
        normalized: list[dict[str, Any]] = []
        for message in input_value:
            if not isinstance(message, dict) or "role" not in message:
                raise ValueError("input must be a string or a list of chat messages")
            normalized.append(_ensure_text_content(message))
        return normalized
    raise ValueError("input must be a string or a list of chat messages")


_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
]


def _strip_thinking(text: str) -> str:
    for pattern in _THINK_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


@dataclass(frozen=True, slots=True)
class FamilyStrategy:
    build_prompt: Callable[[Any, list[dict[str, Any]], bool], str]
    sampling_defaults: Callable[[bool], JsonDict]


def _build_prompt_with_kwargs(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer.apply_chat_template is required for VLLMProvider")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def _qwen_prompt(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    return _build_prompt_with_kwargs(tokenizer, messages, enable_thinking=enable_thinking)


def _gemma_prompt(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    adjusted_messages = [dict(message) for message in messages]
    if enable_thinking:
        for index, message in enumerate(adjusted_messages):
            if message["role"] == "system":
                adjusted_messages[index] = {**message, "content": f"<|think|>{message['content']}"}
                break
        else:
            adjusted_messages.insert(0, {"role": "system", "content": "<|think|>"})
    return _build_prompt_with_kwargs(tokenizer, adjusted_messages)


def _qwen_sampling_defaults(enable_thinking: bool) -> JsonDict:
    if enable_thinking:
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 2048,
        }
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 256,
    }


def _gemma_sampling_defaults(enable_thinking: bool) -> JsonDict:
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 2048 if enable_thinking else 256,
    }


FAMILY_STRATEGIES: dict[str, FamilyStrategy] = {
    "gemma": FamilyStrategy(build_prompt=_gemma_prompt, sampling_defaults=_gemma_sampling_defaults),
    "qwen": FamilyStrategy(build_prompt=_qwen_prompt, sampling_defaults=_qwen_sampling_defaults),
}


def _is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _normalize_metal_memory_fraction(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return value
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError("metal_memory_fraction must be 'auto' or a number in (0, 1]") from exc
    else:
        numeric = float(value)
    if not 0 < numeric <= 1:
        raise ValueError("metal_memory_fraction must be 'auto' or a number in (0, 1]")
    return str(value)


def _missing_vllm_message() -> str:
    if not _is_macos_arm64():
        return "vllm is required for RailClient.vllm(). Install it with `uv add 'tokenrail[vllm]'`."
    if sys.version_info < (3, 12):
        return (
            "vllm-metal on Apple Silicon requires Python 3.12 or newer. "
            "Create a Python 3.12+ environment, then install tokenrail with `uv add 'tokenrail[vllm]'`. "
            "If needed, install vLLM-Metal with "
            "`curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash`."
        )
    return (
        "vllm-metal is required for RailClient.vllm() on Apple Silicon. "
        "Use Python 3.12 or newer, then install tokenrail with `uv add 'tokenrail[vllm]'`, "
        "or install vLLM-Metal with "
        "`curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash`."
    )


class VLLMProvider(BaseProvider):
    name = "vllm"
    supports_batching = True

    def __init__(
        self,
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
        llm: Any | None = None,
        tokenizer: Any | None = None,
        sampling_params_cls: Any | None = None,
    ) -> None:
        if family not in FAMILY_STRATEGIES:
            supported = ", ".join(sorted(FAMILY_STRATEGIES))
            raise ValueError(f"Unsupported vLLM family {family!r}. Supported families: {supported}")
        self.model_id = model_id
        self.family = family
        self.batch_flush_size = max(batch_flush_size, 1)
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.enable_prefix_caching = enable_prefix_caching
        self.seed = seed
        self.device = device
        self.metal_memory_fraction = _normalize_metal_memory_fraction(metal_memory_fraction)
        self.extra_llm_kwargs = dict(extra_llm_kwargs or {})
        self._llm = llm
        self._tokenizer = tokenizer
        self._sampling_params_cls = sampling_params_cls

    @property
    def strategy(self) -> FamilyStrategy:
        return FAMILY_STRATEGIES[self.family]

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        if self._llm is not None and self._tokenizer is None and hasattr(self._llm, "get_tokenizer"):
            self._tokenizer = self._llm.get_tokenizer()

        if self._llm is not None and self._tokenizer is not None and self._sampling_params_cls is not None:
            return self._llm, self._tokenizer, self._sampling_params_cls

        if self.metal_memory_fraction is not None:
            os.environ["VLLM_METAL_MEMORY_FRACTION"] = self.metal_memory_fraction

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(_missing_vllm_message()) from exc

        if self._llm is None:
            llm_kwargs: JsonDict = {
                "model": self.model_id,
                "dtype": self.dtype,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self.trust_remote_code,
                "enable_prefix_caching": self.enable_prefix_caching,
            }
            if self.device is not None:
                llm_kwargs["device"] = self.device
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len
            if self.quantization is not None:
                llm_kwargs["quantization"] = self.quantization
            if self.seed is not None:
                llm_kwargs["seed"] = self.seed
            collisions = sorted(set(llm_kwargs).intersection(self.extra_llm_kwargs))
            if collisions:
                names = ", ".join(collisions)
                raise ValueError(f"extra_llm_kwargs cannot override explicit VLLMProvider options: {names}")
            llm_kwargs.update(self.extra_llm_kwargs)
            self._llm = LLM(**llm_kwargs)

        if self._tokenizer is None:
            self._tokenizer = self._llm.get_tokenizer()
        if self._sampling_params_cls is None:
            self._sampling_params_cls = SamplingParams
        return self._llm, self._tokenizer, self._sampling_params_cls

    def _validate_request(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        verbosity: str | None,
        response_format: JsonDict | None,
        service_tier: str | None,
        store: bool | None,
    ) -> None:
        if model != self.model_id:
            raise ValueError(f"VLLMProvider is bound to model_id={self.model_id}, got {model}")
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is not supported for VLLMProvider")
        if verbosity is not None:
            raise ValueError("verbosity is not supported for VLLMProvider")
        if response_format is not None:
            raise ValueError("response_format is not supported for VLLMProvider")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for VLLMProvider")
        if store is not None:
            raise ValueError("store is not supported for VLLMProvider")

    def _resolve_sampling_config(self, request: dict[str, Any]) -> tuple[bool, JsonDict]:
        enable_thinking = bool(request.get("enable_thinking", False))
        sampling = dict(self.strategy.sampling_defaults(enable_thinking))
        if request.get("temperature") is not None:
            sampling["temperature"] = float(request["temperature"])
        if request.get("top_p") is not None:
            sampling["top_p"] = float(request["top_p"])
        if request.get("top_k") is not None:
            sampling["top_k"] = int(request["top_k"])
        if request.get("max_output_tokens") is not None:
            sampling["max_tokens"] = int(request["max_output_tokens"])
        if self.seed is not None:
            sampling["seed"] = self.seed
        return enable_thinking, sampling

    def _sampling_group_key(self, enable_thinking: bool, sampling: JsonDict) -> tuple[Any, ...]:
        return (
            enable_thinking,
            sampling.get("temperature"),
            sampling.get("top_p"),
            sampling.get("top_k"),
            sampling.get("max_tokens"),
        )

    def _split_group(
        self,
        grouped_requests: list[tuple[dict[str, Any], str, JsonDict]],
    ) -> list[list[tuple[dict[str, Any], str, JsonDict]]]:
        return [
            grouped_requests[index : index + self.batch_flush_size]
            for index in range(0, len(grouped_requests), self.batch_flush_size)
        ]

    def _build_response(
        self,
        *,
        request: dict[str, Any],
        generated: Any,
        output: Any,
        started_at: float,
        completed_at: float,
    ) -> NormalizedResponse:
        prompt_token_ids = list(getattr(generated, "prompt_token_ids", []) or [])
        token_ids = list(getattr(output, "token_ids", []) or [])
        raw_text = str(getattr(output, "text", ""))
        output_text = _strip_thinking(raw_text)
        usage = UsageBreakdown(
            input_tokens=len(prompt_token_ids),
            cached_tokens=0,
            output_tokens=len(token_ids),
            reasoning_tokens=0,
            total_tokens=len(prompt_token_ids) + len(token_ids),
        )
        finish_reason = getattr(output, "finish_reason", None)
        raw_response = {
            "id": request.get("request_id"),
            "object": "local_response",
            "model": self.model_id,
            "provider": self.name,
            "output_text": output_text,
            "usage": usage.to_dict(),
            "finish_reason": finish_reason,
        }
        return NormalizedResponse(
            id=str(request.get("request_id") or ""),
            model=self.model_id,
            provider=self.name,
            output_text=output_text,
            raw_response=raw_response,
            usage=usage,
            billing=None,
            cost=None,
            timing=TimingBreakdown(
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=completed_at - started_at,
            ),
            metadata=request.get("metadata"),
        )

    def create(self, **kwargs: Any) -> NormalizedResponse:
        return self.create_many([kwargs])[0]

    def create_many(self, requests: list[dict[str, Any]]) -> list[NormalizedResponse]:
        llm, tokenizer, sampling_params_cls = self._load_runtime()

        responses_by_id: dict[str, NormalizedResponse] = {}
        grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], str, JsonDict]]] = {}
        order: list[str] = []

        for index, request in enumerate(requests):
            request_id = str(request.get("request_id") or f"local-{index}")
            order.append(request_id)
            request_model = str(request.get("model") or self.model_id)
            self._validate_request(
                model=request_model,
                reasoning_effort=request.get("reasoning_effort"),
                verbosity=request.get("verbosity"),
                response_format=request.get("response_format"),
                service_tier=request.get("service_tier"),
                store=request.get("store"),
            )
            messages = _normalize_messages(request["input"])
            enable_thinking, sampling = self._resolve_sampling_config(request)
            prompt = self.strategy.build_prompt(tokenizer, messages, enable_thinking)
            group_key = self._sampling_group_key(enable_thinking, sampling)
            request_with_id = dict(request)
            request_with_id.setdefault("request_id", request_id)
            grouped.setdefault(group_key, []).append((request_with_id, prompt, sampling))

        for grouped_requests in grouped.values():
            for request_chunk in self._split_group(grouped_requests):
                prompts = [prompt for _, prompt, _ in request_chunk]
                sampling_kwargs = dict(request_chunk[0][2])
                sampling_params = sampling_params_cls(**sampling_kwargs)
                started_at = time.time()
                generated_batch = llm.generate(prompts, sampling_params, use_tqdm=False)
                completed_at = time.time()

                for request, generated in zip((item[0] for item in request_chunk), generated_batch):
                    outputs = list(getattr(generated, "outputs", []) or [])
                    if not outputs:
                        raise RuntimeError("vLLM returned no outputs for request")
                    responses_by_id[str(request["request_id"])] = self._build_response(
                        request=request,
                        generated=generated,
                        output=outputs[0],
                        started_at=started_at,
                        completed_at=completed_at,
                    )

        return [responses_by_id[request_id] for request_id in order]
