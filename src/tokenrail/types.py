from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(slots=True)
class UsageBreakdown:
    """Token usage for a single request, including cached and reasoning tokens."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def empty(cls) -> UsageBreakdown:
        return cls()

    @classmethod
    def from_dict(cls, raw: JsonDict | None) -> UsageBreakdown:
        raw = raw or {}
        input_details = raw.get("input_tokens_details") or {}
        output_details = raw.get("output_tokens_details") or {}
        return cls(
            input_tokens=int(raw.get("input_tokens") or 0),
            cached_tokens=int(input_details.get("cached_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    def finalized(self) -> UsageBreakdown:
        total = self.total_tokens or (self.input_tokens + self.output_tokens)
        return UsageBreakdown(
            input_tokens=self.input_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=total,
        )

    def to_dict(self) -> JsonDict:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens or (self.input_tokens + self.output_tokens),
        }


@dataclass(slots=True)
class CostBreakdown:
    """USD cost of a request split by payer (developer vs. OpenAI-covered)."""

    nominal_usd: float = 0.0
    developer_usd: float = 0.0
    openai_usd: float = 0.0
    payer: str | None = None

    @classmethod
    def none(cls, payer: str | None = None) -> CostBreakdown:
        return cls(payer=payer)

    def to_dict(self) -> JsonDict:
        return {
            "nominal_usd": self.nominal_usd,
            "developer_usd": self.developer_usd,
            "openai_usd": self.openai_usd,
            "payer": self.payer,
        }


@dataclass(slots=True)
class TimingBreakdown:
    started_at: float
    completed_at: float
    latency_seconds: float

    def to_dict(self) -> JsonDict:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(slots=True)
class NormalizedResponse:
    """Provider-agnostic result of a single request.

    ``error`` is set (and ``output_text`` is ``None``) when the request failed;
    ``raw_response`` carries the unmodified provider payload when available.
    """

    id: str
    model: str
    provider: str
    output_text: str | None
    raw_response: JsonDict | None
    usage: UsageBreakdown = field(default_factory=UsageBreakdown.empty)
    billing: JsonDict | None = None
    cost: CostBreakdown | None = None
    timing: TimingBreakdown | None = None
    error: str | None = None
    metadata: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "model": self.model,
            "provider": self.provider,
            "output_text": self.output_text,
            "raw_response": self.raw_response,
            "usage": self.usage.finalized().to_dict(),
            "billing": self.billing,
            "cost": self.cost.to_dict() if self.cost is not None else None,
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class BatchItem:
    """One unit of batch work: a stable id plus ``responses.create`` kwargs."""

    id: str
    request_kwargs: JsonDict


@dataclass(slots=True)
class ModelStats:
    requests: int = 0
    success: int = 0
    errors: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    nominal_usd: float = 0.0
    developer_usd: float = 0.0
    openai_usd: float = 0.0

    def to_dict(self) -> JsonDict:
        return {
            "requests": self.requests,
            "success": self.success,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "nominal_usd": self.nominal_usd,
            "developer_usd": self.developer_usd,
            "openai_usd": self.openai_usd,
        }


@dataclass(slots=True)
class StatsSnapshot:
    """Point-in-time view of batch progress, usage, cost, and ETA."""

    total_requests: int = 0
    todo_requests: int = 0
    processed_requests: int = 0
    success_requests: int = 0
    error_requests: int = 0
    skipped_requests: int = 0
    remaining_requests: int = 0
    started_at: float | None = None
    last_updated_at: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    estimated_finished_at: float | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    rolling_rpm: float = 0.0
    rolling_tpm: float = 0.0
    nominal_usd: float = 0.0
    developer_usd: float = 0.0
    openai_usd: float = 0.0
    by_model: dict[str, ModelStats] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "total_requests": self.total_requests,
            "todo_requests": self.todo_requests,
            "processed_requests": self.processed_requests,
            "success_requests": self.success_requests,
            "error_requests": self.error_requests,
            "skipped_requests": self.skipped_requests,
            "remaining_requests": self.remaining_requests,
            "started_at": self.started_at,
            "last_updated_at": self.last_updated_at,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "estimated_finished_at": self.estimated_finished_at,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "rolling_rpm": self.rolling_rpm,
            "rolling_tpm": self.rolling_tpm,
            "nominal_usd": self.nominal_usd,
            "developer_usd": self.developer_usd,
            "openai_usd": self.openai_usd,
            "by_model": {model: stats.to_dict() for model, stats in self.by_model.items()},
        }
