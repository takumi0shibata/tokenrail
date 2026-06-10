from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Sequence

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


class _SubmitRateLimiter:
    def __init__(
        self,
        *,
        max_rpm: int | None,
        max_tpm: int | None,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_rpm is not None and max_rpm < 1:
            raise ValueError("max_rpm must be at least 1")
        if max_tpm is not None and max_tpm < 1:
            raise ValueError("max_tpm must be at least 1")
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.window_seconds = window_seconds
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._submitted_at: deque[float] = deque()
        self._completed_events: deque[tuple[float, int]] = deque()
        self._inflight_estimates: deque[int] = deque()
        self._inflight_estimated_tokens = 0
        self._completed_requests = 0
        self._completed_tokens = 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._submitted_at and self._submitted_at[0] <= cutoff:
            self._submitted_at.popleft()
        while self._completed_events and self._completed_events[0][0] <= cutoff:
            self._completed_events.popleft()

    def _estimated_next_tokens(self) -> int:
        if self._completed_requests == 0:
            return 0
        return (self._completed_tokens + self._completed_requests - 1) // self._completed_requests

    def _rolling_completed_tokens(self) -> int:
        return sum(tokens for _, tokens in self._completed_events)

    def can_submit(self) -> bool:
        now = self.time_fn()
        self._prune(now)
        if self.max_rpm is not None and len(self._submitted_at) >= self.max_rpm:
            return False
        if self.max_tpm is not None:
            if self._completed_requests == 0 and self._submitted_at:
                return False
            estimated_next = self._estimated_next_tokens()
            if not self._completed_events and self._inflight_estimated_tokens == 0:
                return True
            if self._rolling_completed_tokens() + self._inflight_estimated_tokens + estimated_next > self.max_tpm:
                return False
        return True

    def retry_after(self) -> float | None:
        now = self.time_fn()
        self._prune(now)
        waits: list[float] = []
        if self.max_rpm is not None and len(self._submitted_at) >= self.max_rpm:
            waits.append(self._submitted_at[0] + self.window_seconds - now)
        if self.max_tpm is not None and self._completed_events:
            if self._rolling_completed_tokens() + self._inflight_estimated_tokens + self._estimated_next_tokens() > self.max_tpm:
                waits.append(self._completed_events[0][0] + self.window_seconds - now)
        if waits:
            return max(min(waits), 0.0)
        if not self.can_submit():
            return None
        return 0.0

    def wait_until_allowed(self) -> None:
        while not self.can_submit():
            self.sleep_fn(self.retry_after() or 0.01)

    def record_submit(self) -> None:
        now = self.time_fn()
        self._prune(now)
        self._submitted_at.append(now)
        estimated_tokens = self._estimated_next_tokens() if self.max_tpm is not None else 0
        self._inflight_estimates.append(estimated_tokens)
        self._inflight_estimated_tokens += estimated_tokens

    def record_completion(self, response: NormalizedResponse) -> None:
        now = self.time_fn()
        self._prune(now)
        if self._inflight_estimates:
            self._inflight_estimated_tokens -= self._inflight_estimates.popleft()
        total_tokens = response.usage.total_tokens or (response.usage.input_tokens + response.usage.output_tokens)
        self._completed_events.append((now, total_tokens))
        self._completed_requests += 1
        self._completed_tokens += total_tokens


class BatchExecutor:
    def __init__(
        self,
        *,
        client: Any,
        max_workers: int = 20,
        max_rpm: int | None = None,
        max_tpm: int | None = None,
        sinks: Sequence[ResultSink] | None = None,
        monitor: RollingMetricsMonitor | None = None,
    ) -> None:
        self.client = client
        self.max_workers = max_workers
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.sinks = list(sinks or [])
        self.monitor = monitor or RollingMetricsMonitor()
        self._time_fn = time.time
        self._sleep_fn = time.sleep

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
        limiter = _SubmitRateLimiter(
            max_rpm=self.max_rpm,
            max_tpm=self.max_tpm,
            time_fn=self._time_fn,
            sleep_fn=self._sleep_fn,
        )
        next_item = 0
        pending: set[Future[NormalizedResponse]] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while next_item < len(items) or pending:
                while next_item < len(items) and len(pending) < self.max_workers and limiter.can_submit():
                    limiter.record_submit()
                    pending.add(executor.submit(self._call_single, items[next_item]))
                    next_item += 1

                if not pending:
                    if next_item < len(items):
                        limiter.wait_until_allowed()
                    continue

                timeout = None
                if next_item < len(items) and len(pending) < self.max_workers and not limiter.can_submit():
                    timeout = limiter.retry_after()
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    continue

                for future in done:
                    response = future.result()
                    limiter.record_completion(response)
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
