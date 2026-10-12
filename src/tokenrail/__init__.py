"""Thin client and batch execution helpers for OpenAI Responses API workloads.

tokenrail wraps the OpenAI Responses API with a ``client.responses``-style surface
and adds thread-based batch execution, structured output parsing, client-side
RPM/TPM submit throttling, per-model token/cost monitoring, and resumable JSONL
result writing.
"""

from .client import RailClient
from .executor import BatchExecutor, batch_items_from_queries
from .providers import OpenAIProvider
from .types import BatchItem, CostBreakdown, NormalizedResponse, StatsSnapshot, UsageBreakdown

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "BatchExecutor",
    "BatchItem",
    "CostBreakdown",
    "NormalizedResponse",
    "OpenAIProvider",
    "RailClient",
    "StatsSnapshot",
    "UsageBreakdown",
    "batch_items_from_queries",
]
