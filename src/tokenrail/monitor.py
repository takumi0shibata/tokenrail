from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime

from .types import ModelStats, NormalizedResponse, StatsSnapshot


class RollingMetricsMonitor:
    """Thread-safe aggregator of per-request usage, cost, and throughput metrics.

    Tracks totals and per-model breakdowns, rolling RPM/TPM over
    ``window_seconds``, and ETA estimates. Each :meth:`record` call prints a
    one-line progress update via ``printer`` (pass ``printer=None`` to disable).
    """

    def __init__(
        self,
        *,
        window_seconds: int = 60,
        printer: Callable[[str], None] | None = print,
    ) -> None:
        self.window_seconds = window_seconds
        self.printer = printer
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()
        self._snapshot = StatsSnapshot()

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._snapshot = StatsSnapshot()

    def start(self, *, total_requests: int, todo_requests: int, skipped_requests: int) -> StatsSnapshot:
        with self._lock:
            now = time.time()
            self._snapshot.total_requests = total_requests
            self._snapshot.todo_requests = todo_requests
            self._snapshot.skipped_requests = skipped_requests
            self._snapshot.remaining_requests = todo_requests
            self._snapshot.started_at = now
            self._snapshot.last_updated_at = now
            self._snapshot.elapsed_seconds = 0.0
            if todo_requests == 0:
                self._snapshot.eta_seconds = 0.0
                self._snapshot.estimated_finished_at = now
            else:
                self._snapshot.eta_seconds = None
                self._snapshot.estimated_finished_at = None
            return self._copy_snapshot_unlocked()

    def _recompute_time_fields_unlocked(self, now: float) -> None:
        self._snapshot.last_updated_at = now
        if self._snapshot.started_at is None:
            self._snapshot.elapsed_seconds = 0.0
            self._snapshot.remaining_requests = self._snapshot.todo_requests
            self._snapshot.eta_seconds = None
            self._snapshot.estimated_finished_at = None
            return

        self._snapshot.elapsed_seconds = max(now - self._snapshot.started_at, 0.0)
        self._snapshot.remaining_requests = max(self._snapshot.todo_requests - self._snapshot.processed_requests, 0)
        if self._snapshot.todo_requests == 0:
            self._snapshot.eta_seconds = 0.0
            self._snapshot.estimated_finished_at = self._snapshot.started_at
            return
        if self._snapshot.processed_requests == 0 or self._snapshot.elapsed_seconds <= 0:
            self._snapshot.eta_seconds = None
            self._snapshot.estimated_finished_at = None
            return

        processing_rate = self._snapshot.processed_requests / self._snapshot.elapsed_seconds
        if processing_rate <= 0:
            self._snapshot.eta_seconds = None
            self._snapshot.estimated_finished_at = None
            return

        eta_seconds = self._snapshot.remaining_requests / processing_rate
        self._snapshot.eta_seconds = eta_seconds
        self._snapshot.estimated_finished_at = now + eta_seconds

    def _copy_snapshot_unlocked(self) -> StatsSnapshot:
        return StatsSnapshot(
            total_requests=self._snapshot.total_requests,
            todo_requests=self._snapshot.todo_requests,
            processed_requests=self._snapshot.processed_requests,
            success_requests=self._snapshot.success_requests,
            error_requests=self._snapshot.error_requests,
            skipped_requests=self._snapshot.skipped_requests,
            remaining_requests=self._snapshot.remaining_requests,
            started_at=self._snapshot.started_at,
            last_updated_at=self._snapshot.last_updated_at,
            elapsed_seconds=self._snapshot.elapsed_seconds,
            eta_seconds=self._snapshot.eta_seconds,
            estimated_finished_at=self._snapshot.estimated_finished_at,
            input_tokens=self._snapshot.input_tokens,
            cached_tokens=self._snapshot.cached_tokens,
            output_tokens=self._snapshot.output_tokens,
            reasoning_tokens=self._snapshot.reasoning_tokens,
            total_tokens=self._snapshot.total_tokens,
            rolling_rpm=self._snapshot.rolling_rpm,
            rolling_tpm=self._snapshot.rolling_tpm,
            nominal_usd=self._snapshot.nominal_usd,
            developer_usd=self._snapshot.developer_usd,
            openai_usd=self._snapshot.openai_usd,
            by_model={model: ModelStats(**stats.to_dict()) for model, stats in self._snapshot.by_model.items()},
        )

    def record(self, response: NormalizedResponse) -> StatsSnapshot:
        """Fold ``response`` into the running totals and return a snapshot copy."""
        with self._lock:
            now = time.time()
            total_tokens = response.usage.total_tokens or (response.usage.input_tokens + response.usage.output_tokens)
            self._events.append((now, total_tokens))
            cutoff = now - self.window_seconds
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

            model_stats = self._snapshot.by_model.setdefault(response.model, ModelStats())
            model_stats.requests += 1
            self._snapshot.processed_requests += 1

            if response.error is None:
                self._snapshot.success_requests += 1
                model_stats.success += 1
            else:
                self._snapshot.error_requests += 1
                model_stats.errors += 1

            self._snapshot.input_tokens += response.usage.input_tokens
            self._snapshot.cached_tokens += response.usage.cached_tokens
            self._snapshot.output_tokens += response.usage.output_tokens
            self._snapshot.reasoning_tokens += response.usage.reasoning_tokens
            self._snapshot.total_tokens += total_tokens

            model_stats.input_tokens += response.usage.input_tokens
            model_stats.cached_tokens += response.usage.cached_tokens
            model_stats.output_tokens += response.usage.output_tokens
            model_stats.reasoning_tokens += response.usage.reasoning_tokens
            model_stats.total_tokens += total_tokens

            if response.cost is not None:
                self._snapshot.nominal_usd += response.cost.nominal_usd
                self._snapshot.developer_usd += response.cost.developer_usd
                self._snapshot.openai_usd += response.cost.openai_usd
                model_stats.nominal_usd += response.cost.nominal_usd
                model_stats.developer_usd += response.cost.developer_usd
                model_stats.openai_usd += response.cost.openai_usd

            self._snapshot.rolling_rpm = float(len(self._events))
            self._snapshot.rolling_tpm = float(sum(tokens for _, tokens in self._events))
            self._recompute_time_fields_unlocked(now)
            snapshot = self._copy_snapshot_unlocked()

        if self.printer is not None:
            self.printer(self.format_update(response, snapshot))
        return snapshot

    def snapshot(self) -> StatsSnapshot:
        """Return a copy of the current stats without recording anything."""
        with self._lock:
            return self._copy_snapshot_unlocked()

    def finalize(self, *, total_requests: int, skipped_requests: int) -> StatsSnapshot:
        with self._lock:
            self._snapshot.total_requests = total_requests
            self._snapshot.skipped_requests = skipped_requests
            self._snapshot.todo_requests = max(total_requests - skipped_requests, 0)
            self._recompute_time_fields_unlocked(time.time())
            return self._copy_snapshot_unlocked()

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None:
            return "-"
        whole = max(int(round(seconds)), 0)
        hours, remainder = divmod(whole, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_finish_time(self, timestamp: float | None) -> str:
        if timestamp is None:
            return "-"
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    def format_update(self, response: NormalizedResponse, snapshot: StatsSnapshot) -> str:
        usage = response.usage
        payer = response.cost.payer if response.cost is not None else None
        nominal = response.cost.nominal_usd if response.cost is not None else 0.0
        return (
            f"[{snapshot.processed_requests}/{snapshot.total_requests}] id={response.id} model={response.model} "
            f"status={'ok' if response.error is None else 'error'} "
            f"elapsed={self._format_duration(snapshot.elapsed_seconds)} "
            f"eta={self._format_duration(snapshot.eta_seconds)} "
            f"finish={self._format_finish_time(snapshot.estimated_finished_at)} "
            f"in={usage.input_tokens} cached={usage.cached_tokens} out={usage.output_tokens} "
            f"reasoning={usage.reasoning_tokens} total={usage.total_tokens} "
            f"rpm={snapshot.rolling_rpm:.0f} tpm={snapshot.rolling_tpm:.0f} "
            f"cost=${nominal:.6f} payer={payer or '-'} "
            f"developer_total=${snapshot.developer_usd:.6f} openai_total=${snapshot.openai_usd:.6f}"
        )
