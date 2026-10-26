"""Thin client and batch execution helpers for Responses and Decisions workloads.

tokenrail wraps OpenAI Responses and Decisions with SDK-style namespaces
and adds thread-based batch execution, structured output parsing, client-side
RPM/TPM submit throttling, per-model token/cost monitoring, and resumable JSONL
result writing.
"""

from importlib.metadata import version as _distribution_version

from .client import RailClient
from .executor import BatchExecutor, batch_items_from_queries
from .providers import OpenAIProvider
from .types import BatchItem, CostBreakdown, NormalizedResponse, StatsSnapshot, UsageBreakdown

__version__ = _distribution_version("tokenrail")

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
