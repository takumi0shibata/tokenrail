from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Literal

from .monitor import RollingMetricsMonitor
from .prompt_cache import PromptCacheConfig, build_prompt_cache_plan
from .sinks import ResultSink
from .types import BatchItem, NormalizedResponse, StatsSnapshot, TimingBreakdown, UsageBreakdown


def batch_items_from_queries(queries: dict[str, Any], **shared_request_kwargs: Any) -> list[BatchItem]:
    """Build :class:`BatchItem` objects from an ``{id: input}`` mapping.

    Each value becomes the request ``input``; ``shared_request_kwargs`` (e.g.
    ``model``, ``reasoning_effort``) are applied to every item.
    """
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
            projected = (
                self._rolling_completed_tokens() + self._inflight_estimated_tokens + self._estimated_next_tokens()
            )
            if projected > self.max_tpm:
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


class _PerKeySubmitRateLimiter:
    def __init__(
        self,
        *,
        max_rpm_per_key: int,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if max_rpm_per_key < 1:
            raise ValueError("max_rpm_per_key must be at least 1")
        self.max_rpm_per_key = max_rpm_per_key
        self.window_seconds = window_seconds
        self.time_fn = time_fn
        self._submitted_by_key: dict[str, deque[float]] = {}

    def _events(self, key: str, now: float) -> deque[float]:
        events = self._submitted_by_key.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def can_submit(self, key: str) -> bool:
        return len(self._events(key, self.time_fn())) < self.max_rpm_per_key

    def retry_after(self, key: str) -> float:
        now = self.time_fn()
        events = self._events(key, now)
        if len(events) < self.max_rpm_per_key:
            return 0.0
        return max(events[0] + self.window_seconds - now, 0.0)

    def record_submit(self, key: str) -> None:
        now = self.time_fn()
        self._events(key, now).append(now)


class BatchExecutor:
    """Thread-based batch runner for :class:`~tokenrail.client.RailClient` requests.

    Submits items to a thread pool while honoring optional client-side
    ``max_rpm`` / ``max_tpm`` submit limits, writes each result to the
    configured sinks, and records metrics on the monitor. Items whose ids are
    already present in the first sink are skipped, which makes re-runs
    resumable. Request errors are captured as error responses rather than
    raised, so a single failing item does not abort the batch.
    """

    def __init__(
        self,
        *,
        client: Any,
        max_workers: int = 20,
        max_rpm: int | None = None,
        max_tpm: int | None = None,
        prompt_cache: Literal["auto"] | PromptCacheConfig | None = None,
        sinks: Sequence[ResultSink] | None = None,
        monitor: RollingMetricsMonitor | None = None,
    ) -> None:
        self.client = client
        self.max_workers = max_workers
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        if prompt_cache == "auto":
            self.prompt_cache = PromptCacheConfig()
        elif prompt_cache is None or isinstance(prompt_cache, PromptCacheConfig):
            self.prompt_cache = prompt_cache
        else:
            raise ValueError("prompt_cache must be None, 'auto', or PromptCacheConfig")
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
            if request_kwargs.get("text_format") is not None:
                return self.client.responses.parse(**request_kwargs)
            return self.client.responses.create(**request_kwargs)
        except Exception as exc:
            model = str(request_kwargs.get("model") or getattr(self.client.provider, "model_id", "unknown"))
            return _error_response(item.id, model=model, provider=self.client.provider.name, error=exc)

    def _run_threaded(self, items: list[BatchItem], *, cache_target_rpm_per_shard: int | None = None) -> None:
        if cache_target_rpm_per_shard is not None:
            self._run_threaded_sharded(items, cache_target_rpm_per_shard=cache_target_rpm_per_shard)
            return

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

    def _run_threaded_sharded(self, items: list[BatchItem], *, cache_target_rpm_per_shard: int) -> None:
        limiter = _SubmitRateLimiter(
            max_rpm=self.max_rpm,
            max_tpm=self.max_tpm,
            time_fn=self._time_fn,
            sleep_fn=self._sleep_fn,
        )
        shard_limiter = _PerKeySubmitRateLimiter(
            max_rpm_per_key=cache_target_rpm_per_shard,
            time_fn=self._time_fn,
        )
        remaining = list(items)
        pending: set[Future[NormalizedResponse]] = set()

        def cache_key(item: BatchItem) -> str:
            value = item.request_kwargs.get("prompt_cache_key")
            if not isinstance(value, str) or not value:
                raise ValueError("planned prompt-cache item is missing prompt_cache_key")
            return value

        def next_delay() -> float | None:
            if not remaining:
                return None
            if limiter.can_submit():
                global_delay: float | None = 0.0
            else:
                global_delay = limiter.retry_after()
            shard_delay = min(shard_limiter.retry_after(cache_key(item)) for item in remaining)
            if global_delay is None:
                return None
            return max(global_delay, shard_delay)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while remaining or pending:
                while remaining and len(pending) < self.max_workers and limiter.can_submit():
                    selected_index = next(
                        (
                            index
                            for index, item in enumerate(remaining)
                            if shard_limiter.can_submit(cache_key(item))
                        ),
                        None,
                    )
                    if selected_index is None:
                        break
                    item = remaining.pop(selected_index)
                    key = cache_key(item)
                    limiter.record_submit()
                    shard_limiter.record_submit(key)
                    pending.add(executor.submit(self._call_single, item))

                if not pending:
                    if remaining:
                        self._sleep_fn(next_delay() or 0.01)
                    continue

                timeout = None
                if remaining and len(pending) < self.max_workers:
                    timeout = next_delay()
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    continue

                for future in done:
                    response = future.result()
                    limiter.record_completion(response)
                    self._save(response)
                    self.monitor.record(response)

    def run(self, items: Sequence[BatchItem] | dict[str, Any]) -> StatsSnapshot:
        """Execute ``items`` (a sequence of :class:`BatchItem` or an ``{id: input}``
        dict) and return the final :class:`~tokenrail.types.StatsSnapshot`."""
        self.monitor.reset()
        normalized_items = self._prepare_items(items)
        cache_shards = 0
        cache_target_rpm_per_shard: int | None = None
        if self.prompt_cache is not None:
            provider_name = str(getattr(getattr(self.client, "provider", None), "name", ""))
            cache_plan = build_prompt_cache_plan(
                normalized_items,
                config=self.prompt_cache,
                max_rpm=self.max_rpm,
                provider_name=provider_name,
            )
            normalized_items = cache_plan.items
            cache_shards = cache_plan.num_shards
            cache_target_rpm_per_shard = cache_plan.target_rpm_per_shard
        done_ids = self._load_done_ids()
        todo = [item for item in normalized_items if item.id not in done_ids]
        skipped = len(normalized_items) - len(todo)
        self.monitor.start(
            total_requests=len(normalized_items),
            todo_requests=len(todo),
            skipped_requests=skipped,
            prompt_cache_shards=cache_shards,
            prompt_cache_target_rpm_per_shard=cache_target_rpm_per_shard,
        )

        self._run_threaded(todo, cache_target_rpm_per_shard=cache_target_rpm_per_shard)

        return self.monitor.finalize(total_requests=len(normalized_items), skipped_requests=skipped)
