from __future__ import annotations

import time
from typing import Any

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
                raise ValueError("HFTransformersProvider only supports text content in v1")
            item_type = item.get("type")
            if item_type in {"input_text", "text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
                continue
            raise ValueError("HFTransformersProvider only supports text content in v1")
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


class HFTransformersProvider(BaseProvider):
    name = "hf"
    supports_batching = True

    def __init__(
        self,
        *,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.batch_size = max(batch_size, 1)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._model, self._tokenizer, self._torch

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for RailClient.hf(). Install them with `uv add 'tokenrail[hf]'`."
            ) from exc

        torch_dtype = getattr(torch, self.dtype) if self.dtype not in {"auto", ""} else "auto"
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model.eval()
        return self._model, self._tokenizer, self._torch

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
            raise ValueError(f"HFTransformersProvider is bound to model_id={self.model_id}, got {model}")
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is not supported for HFTransformersProvider")
        if verbosity is not None:
            raise ValueError("verbosity is not supported for HFTransformersProvider")
        if response_format is not None:
            raise ValueError("response_format is not supported for HFTransformersProvider")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for HFTransformersProvider")
        if store is not None:
            raise ValueError("store is not supported for HFTransformersProvider")

    def create(self, **kwargs: Any) -> NormalizedResponse:
        return self.create_many([kwargs])[0]

    def create_many(self, requests: list[dict[str, Any]]) -> list[NormalizedResponse]:
        model, tokenizer, torch = self._load_model()

        results: list[NormalizedResponse] = []
        for index in range(0, len(requests), self.batch_size):
            chunk = requests[index : index + self.batch_size]
            grouped: dict[tuple[float | None, float | None, int], list[dict[str, Any]]] = {}
            for request in chunk:
                request_model = str(request.get("model") or self.model_id)
                self._validate_request(
                    model=request_model,
                    reasoning_effort=request.get("reasoning_effort"),
                    verbosity=request.get("verbosity"),
                    response_format=request.get("response_format"),
                    service_tier=request.get("service_tier"),
                    store=request.get("store"),
                )
                max_output_tokens = int(request.get("max_output_tokens") or 256)
                key = (request.get("temperature"), request.get("top_p"), max_output_tokens)
                grouped.setdefault(key, []).append(request)

            for (temperature, top_p, max_output_tokens), grouped_requests in grouped.items():
                started_at = time.time()
                messages_batch = [_normalize_messages(request["input"]) for request in grouped_requests]
                prompts: list[str] = []
                for messages in messages_batch:
                    if hasattr(tokenizer, "apply_chat_template"):
                        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                    else:
                        prompts.append("\n".join(f"{message['role']}: {message['content']}" for message in messages))

                tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
                tokenized = {name: tensor.to(model.device) for name, tensor in tokenized.items()}
                input_lengths = tokenized["attention_mask"].sum(dim=1)
                generation_kwargs: JsonDict = {
                    "max_new_tokens": max_output_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if temperature is not None:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["do_sample"] = temperature > 0
                else:
                    generation_kwargs["do_sample"] = False
                if top_p is not None:
                    generation_kwargs["top_p"] = top_p
                    generation_kwargs["do_sample"] = True

                with torch.inference_mode():
                    sequences = model.generate(**tokenized, **generation_kwargs)

                completed_at = time.time()
                for row, request in enumerate(grouped_requests):
                    input_tokens = int(input_lengths[row].item())
                    output_tokens = int(sequences[row].shape[-1] - input_tokens)
                    output_ids = sequences[row][input_tokens:]
                    text = tokenizer.decode(output_ids, skip_special_tokens=True)
                    usage = UsageBreakdown(
                        input_tokens=input_tokens,
                        cached_tokens=0,
                        output_tokens=output_tokens,
                        reasoning_tokens=0,
                        total_tokens=input_tokens + output_tokens,
                    )
                    raw_response = {
                        "id": request.get("request_id"),
                        "object": "local_response",
                        "model": self.model_id,
                        "provider": self.name,
                        "output_text": text,
                        "usage": usage.to_dict(),
                    }
                    results.append(
                        NormalizedResponse(
                            id=str(request.get("request_id") or f"local-{index + row}"),
                            model=self.model_id,
                            provider=self.name,
                            output_text=text,
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
                    )
        return results
