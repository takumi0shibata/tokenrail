"""Thin client and batch execution helpers for OpenAI Responses API workloads.

tokenrail wraps the OpenAI Responses API with a ``client.responses.create(...)``-style
surface and adds thread-based batch execution, client-side RPM/TPM submit throttling,
per-model token/cost monitoring, and resumable JSONL / per-request result writing.
"""

from .client import RailClient
from .executor import BatchExecutor, batch_items_from_queries
from .monitor import RollingMetricsMonitor
from .providers import OpenAIProvider
from .sinks import PerRequestJsonSink, ResultsJsonlSink
from .types import BatchItem, CostBreakdown, NormalizedResponse, StatsSnapshot, UsageBreakdown

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BatchExecutor",
    "BatchItem",
    "CostBreakdown",
    "NormalizedResponse",
    "OpenAIProvider",
    "PerRequestJsonSink",
    "RailClient",
    "ResultsJsonlSink",
    "RollingMetricsMonitor",
    "StatsSnapshot",
    "UsageBreakdown",
    "batch_items_from_queries",
]
