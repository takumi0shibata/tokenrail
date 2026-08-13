from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

from .catalog import get_model_capabilities
from .types import BatchItem

_ALLOWED_VARIABLE_REQUEST_FIELDS = {"input", "metadata", "request_id", "safety_identifier"}
_STATEFUL_REQUEST_FIELDS = {"conversation", "previous_response_id", "prompt"}
_SUPPORTED_BREAKPOINT_TYPES = {"input_file", "input_image", "input_text"}


@dataclass(frozen=True, slots=True)
class PromptCacheConfig:
    """Configuration for explicit prompt caching and deterministic key sharding."""

    base_key: str | None = None
    shards: Literal["auto"] | int = "auto"
    expected_rpm: int | None = None
    target_rpm_per_shard: int = 15

    def __post_init__(self) -> None:
        if self.base_key is not None and (not isinstance(self.base_key, str) or not self.base_key):
            raise ValueError("prompt cache base_key must not be empty")
        if self.shards != "auto" and (
            not isinstance(self.shards, int) or isinstance(self.shards, bool) or self.shards < 1
        ):
            raise ValueError("prompt cache shards must be 'auto' or an integer of at least 1")
        if self.expected_rpm is not None and (
            not isinstance(self.expected_rpm, int) or isinstance(self.expected_rpm, bool) or self.expected_rpm < 1
        ):
            raise ValueError("prompt cache expected_rpm must be at least 1")
        if (
            not isinstance(self.target_rpm_per_shard, int)
            or isinstance(self.target_rpm_per_shard, bool)
            or self.target_rpm_per_shard < 1
        ):
            raise ValueError("prompt cache target_rpm_per_shard must be at least 1")


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    items: list[BatchItem]
    base_key: str
    num_shards: int
    target_rpm_per_shard: int


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, type):
        identity: dict[str, Any] = {
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
        if hasattr(value, "model_json_schema"):
            identity["schema"] = _canonicalize(value.model_json_schema())
        elif hasattr(value, "schema"):
            identity["schema"] = _canonicalize(value.schema())
        return {"__python_type__": identity}
    if hasattr(value, "model_dump"):
        return _canonicalize(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _canonicalize(value.dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    raise TypeError(f"cannot create a stable prompt cache key from {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _contains_breakpoint(value: Any) -> bool:
    if isinstance(value, dict):
        if "prompt_cache_breakpoint" in value:
            return True
        return any(_contains_breakpoint(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_breakpoint(item) for item in value)
    return False


def _normalize_content(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("prompt_cache='auto' requires message content to be a string or list")
    normalized: list[Any] = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "input_text", "text": block})
        else:
            normalized.append(copy.deepcopy(block))
    return normalized


def _normalize_input(input_value: Any, instructions: Any) -> list[Any]:
    if isinstance(input_value, str):
        normalized: list[Any] = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    elif isinstance(input_value, list):
        normalized = copy.deepcopy(input_value)
        for index, item in enumerate(normalized):
            if isinstance(item, dict) and "role" in item and "content" in item:
                item = dict(item)
                item["content"] = _normalize_content(item["content"])
                normalized[index] = item
    else:
        raise ValueError("prompt_cache='auto' requires Responses input to be a string or list")

    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("prompt_cache='auto' requires top-level instructions to be a string")
        normalized.insert(
            0,
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            },
        )
    return normalized


def _message_parts(item: Any) -> tuple[dict[str, Any], list[Any]] | None:
    if not isinstance(item, dict) or "role" not in item or "content" not in item:
        return None
    envelope = {key: value for key, value in item.items() if key != "content"}
    content = item["content"]
    if not isinstance(content, list):
        return None
    return envelope, content


def _common_breakpoint(inputs: list[list[Any]]) -> tuple[int, int]:
    candidate: tuple[int, int] | None = None
    for item_index in range(min(len(input_items) for input_items in inputs)):
        current = [input_items[item_index] for input_items in inputs]
        message_parts = [_message_parts(item) for item in current]
        if all(parts is not None for parts in message_parts):
            typed_parts = [parts for parts in message_parts if parts is not None]
            envelopes = [parts[0] for parts in typed_parts]
            if any(envelope != envelopes[0] for envelope in envelopes[1:]):
                break
            contents = [parts[1] for parts in typed_parts]
            all_content_equal = all(content == contents[0] for content in contents[1:])
            for content_index in range(min(len(content) for content in contents)):
                blocks = [content[content_index] for content in contents]
                if any(block != blocks[0] for block in blocks[1:]):
                    return _require_breakpoint(candidate)
                block = blocks[0]
                if isinstance(block, dict) and block.get("type") in _SUPPORTED_BREAKPOINT_TYPES:
                    candidate = (item_index, content_index)
            if not all_content_equal:
                break
            continue

        if any(item != current[0] for item in current[1:]):
            break

    return _require_breakpoint(candidate)


def _require_breakpoint(candidate: tuple[int, int] | None) -> tuple[int, int]:
    if candidate is None:
        raise ValueError("prompt_cache='auto' found no common cacheable input content block")
    return candidate


def _prefix_through_breakpoint(input_items: list[Any], breakpoint: tuple[int, int]) -> list[Any]:
    item_index, content_index = breakpoint
    prefix = copy.deepcopy(input_items[: item_index + 1])
    message = dict(prefix[-1])
    message["content"] = copy.deepcopy(message["content"][: content_index + 1])
    prefix[-1] = message
    return prefix


def _mark_breakpoint(input_items: list[Any], breakpoint: tuple[int, int]) -> None:
    item_index, content_index = breakpoint
    block = dict(input_items[item_index]["content"][content_index])
    block["prompt_cache_breakpoint"] = {"mode": "explicit"}
    input_items[item_index]["content"][content_index] = block


def _request_signature(kwargs: dict[str, Any]) -> str:
    signature = {
        key: value
        for key, value in kwargs.items()
        if key not in _ALLOWED_VARIABLE_REQUEST_FIELDS
        and key not in {"instructions", "prompt_cache_key", "prompt_cache_options"}
    }
    return _canonical_json(signature)


def _cache_fingerprint(model: str, kwargs: dict[str, Any], prefix: list[Any]) -> str:
    value = {
        "model": model,
        "tools": kwargs.get("tools"),
        "text": kwargs.get("text"),
        "text_format": kwargs.get("text_format"),
        "response_format": kwargs.get("response_format"),
        "prefix": prefix,
    }
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:24]


def _resolve_num_shards(config: PromptCacheConfig, max_rpm: int | None) -> int:
    if config.shards != "auto":
        return config.shards
    expected_rpm = config.expected_rpm if config.expected_rpm is not None else max_rpm
    if expected_rpm is None:
        raise ValueError("prompt_cache='auto' requires max_rpm or PromptCacheConfig.expected_rpm")
    return max(1, math.ceil(expected_rpm / config.target_rpm_per_shard))


def build_prompt_cache_plan(
    items: list[BatchItem],
    *,
    config: PromptCacheConfig,
    max_rpm: int | None,
    provider_name: str,
) -> PromptCachePlan:
    """Validate, normalize, and assign explicit-cache keys to one homogeneous batch."""
    if provider_name != "openai":
        raise ValueError("prompt_cache='auto' is only supported by the OpenAI provider")
    if not items:
        raise ValueError("prompt_cache='auto' requires at least one batch item")

    normalized_items: list[BatchItem] = []
    existing_keys: list[str | None] = []
    signatures: list[str] = []
    models: list[str] = []

    for item in items:
        kwargs = copy.deepcopy(item.request_kwargs)
        for field in _STATEFUL_REQUEST_FIELDS:
            if kwargs.get(field) is not None:
                raise ValueError(f"prompt_cache='auto' does not support stateful request field {field!r}")
        if kwargs.get("prompt_cache_retention") is not None:
            raise ValueError("prompt_cache='auto' does not support prompt_cache_retention")
        extra_body = kwargs.get("extra_body")
        if kwargs.get("prompt_cache_options") is not None or (
            isinstance(extra_body, dict) and extra_body.get("prompt_cache_options") is not None
        ):
            raise ValueError("prompt_cache='auto' conflicts with existing prompt_cache_options")
        if _contains_breakpoint(kwargs.get("input")):
            raise ValueError("prompt_cache='auto' conflicts with existing prompt_cache_breakpoint")

        model = kwargs.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("prompt_cache='auto' requires every item to specify a model")
        if not get_model_capabilities(model).prompt_cache_explicit:
            raise ValueError(f"explicit prompt caching is not supported for model {model!r}")

        existing_key = kwargs.pop("prompt_cache_key", None)
        if existing_key is not None and (not isinstance(existing_key, str) or not existing_key):
            raise ValueError("prompt_cache_key must be a non-empty string")
        existing_keys.append(existing_key)
        models.append(model)
        signatures.append(_request_signature(kwargs))

        instructions = kwargs.pop("instructions", None)
        kwargs["input"] = _normalize_input(kwargs.get("input"), instructions)
        normalized_items.append(BatchItem(id=str(item.id), request_kwargs=kwargs))

    if any(model != models[0] for model in models[1:]) or any(
        signature != signatures[0] for signature in signatures[1:]
    ):
        raise ValueError("prompt_cache='auto' requires one homogeneous model and request configuration")
    if any(key != existing_keys[0] for key in existing_keys[1:]):
        raise ValueError("prompt_cache='auto' requires the same existing prompt_cache_key on every item")

    normalized_inputs = [item.request_kwargs["input"] for item in normalized_items]
    breakpoint = _common_breakpoint(normalized_inputs)
    prefix = _prefix_through_breakpoint(normalized_inputs[0], breakpoint)

    existing_base_key = existing_keys[0]
    if config.base_key is not None and existing_base_key is not None and config.base_key != existing_base_key:
        raise ValueError("PromptCacheConfig.base_key conflicts with the existing prompt_cache_key")
    base_key = config.base_key or existing_base_key
    if base_key is None:
        base_key = f"tokenrail:{_cache_fingerprint(models[0], normalized_items[0].request_kwargs, prefix)}"

    num_shards = _resolve_num_shards(config, max_rpm)
    planned_items: list[BatchItem] = []
    for item in normalized_items:
        kwargs = copy.deepcopy(item.request_kwargs)
        _mark_breakpoint(kwargs["input"], breakpoint)
        digest = hashlib.sha256(f"{base_key}\0{item.id}".encode()).digest()
        shard_id = int.from_bytes(digest, byteorder="big") % num_shards
        kwargs["prompt_cache_key"] = f"{base_key}:shard-{shard_id}"
        kwargs["prompt_cache_options"] = {"mode": "explicit"}
        planned_items.append(BatchItem(id=item.id, request_kwargs=kwargs))

    return PromptCachePlan(
        items=planned_items,
        base_key=base_key,
        num_shards=num_shards,
        target_rpm_per_shard=config.target_rpm_per_shard,
    )
