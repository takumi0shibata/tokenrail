from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from .monitor import RollingMetricsMonitor
from .sinks import ResultSink
from .types import BatchItem, NormalizedResponse, StatsSnapshot, TimingBreakdown, UsageBreakdown


def batch_items_from_queries(queries: dict[str, Any], **shared_request_kwargs: Any) -> list[BatchItem]:
    return [
        BatchItem(id=str(item_id), request_kwargs={"input": messages, **shared_request_kwargs})
        for item_id, messages in queries.items()
    ]


def _error_response(item_id: str, model: str, provider: str, error: Exception) -> NormalizedResponse:
    return NormalizedResponse(
        id=item_id,
        model=model,
        provider=provider,
        output_text=None,
        raw_response=None,
        usage=UsageBreakdown.empty(),
        billing=None,
        cost=None,
        timing=TimingBreakdown(started_at=0.0, completed_at=0.0, latency_seconds=0.0),
        error=f"{type(error).__name__}: {error}",
    )


class BatchExecutor:
    def __init__(
        self,
        *,
        client: Any,
        max_workers: int = 20,
        sinks: Sequence[ResultSink] | None = None,
        monitor: RollingMetricsMonitor | None = None,
    ) -> None:
        self.client = client
        self.max_workers = max_workers
        self.sinks = list(sinks or [])
        self.monitor = monitor or RollingMetricsMonitor()

    def _save(self, response: NormalizedResponse) -> None:
        for sink in self.sinks:
            sink.save(response)

    def _load_done_ids(self) -> set[str]:
        if not self.sinks:
            return set()
        return self.sinks[0].load_done_ids()

    def _prepare_items(self, items: Sequence[BatchItem] | dict[str, Any]) -> list[BatchItem]:
        if isinstance(items, dict):
            return batch_items_from_queries(items)
        return [BatchItem(id=str(item.id), request_kwargs=dict(item.request_kwargs)) for item in items]

    def _request_kwargs(self, item: BatchItem) -> dict[str, Any]:
        request_kwargs = dict(item.request_kwargs)
        request_kwargs.setdefault("request_id", item.id)
        return request_kwargs

    def _call_single(self, item: BatchItem) -> NormalizedResponse:
        request_kwargs = self._request_kwargs(item)
        try:
            return self.client.responses.create(**request_kwargs)
        except Exception as exc:
            model = str(request_kwargs.get("model") or getattr(self.client.provider, "model_id", "unknown"))
            return _error_response(item.id, model=model, provider=self.client.provider.name, error=exc)

    def _run_threaded(self, items: list[BatchItem]) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._call_single, item) for item in items]
            for future in as_completed(futures):
                response = future.result()
                self._save(response)
                self.monitor.record(response)

    def run(self, items: Sequence[BatchItem] | dict[str, Any]) -> StatsSnapshot:
        self.monitor.reset()
        normalized_items = self._prepare_items(items)
        done_ids = self._load_done_ids()
        todo = [item for item in normalized_items if item.id not in done_ids]
        skipped = len(normalized_items) - len(todo)
        self.monitor.start(
            total_requests=len(normalized_items),
            todo_requests=len(todo),
            skipped_requests=skipped,
        )

        self._run_threaded(todo)

        return self.monitor.finalize(total_requests=len(normalized_items), skipped_requests=skipped)
