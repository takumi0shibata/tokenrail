from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime

from .types import ModelStats, NormalizedResponse, StatsSnapshot

_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"
_KNOWN_PAYERS = {"openai", "developer"}


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        value, suffix = tokens / 1_000_000_000, "B"
    elif tokens >= 1_000_000:
        value, suffix = tokens / 1_000_000, "M"
    elif tokens >= 1_000:
        value, suffix = tokens / 1_000, "k"
    else:
        return str(tokens)
    return f"{value:.1f}".rstrip("0").rstrip(".") + suffix


def _format_usd(value: float, *, digits: int) -> str:
    return f"${value:.{digits}f}"


class RollingMetricsMonitor:
    """Thread-safe aggregator of per-request usage, cost, and throughput metrics.

    The default output separates request details from periodic aggregate
    summaries and highlights payer changes. Pass ``verbose=True`` to retain the
    legacy one-line-per-request format, or ``printer=None`` to disable output.
    """

    def __init__(
        self,
        *,
        window_seconds: int = 60,
        printer: Callable[[str], None] | None = print,
        summary_every: int = 50,
        summary_interval: float | None = 30.0,
        payer_switch_threshold: int = 3,
        verbose: bool = False,
        color: bool | None = None,
    ) -> None:
        if summary_every < 1:
            raise ValueError("summary_every must be at least 1")
        if summary_interval is not None and summary_interval <= 0:
            raise ValueError("summary_interval must be positive or None")
        if payer_switch_threshold < 1:
            raise ValueError("payer_switch_threshold must be at least 1")

        self.window_seconds = window_seconds
        self.printer = printer
        self.summary_every = summary_every
        self.summary_interval = summary_interval
        self.payer_switch_threshold = payer_switch_threshold
        self.verbose = verbose
        self.color = color
        self._color_enabled = self._resolve_color_enabled()
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()
        self._snapshot = StatsSnapshot()
        self._reset_display_state_unlocked()

    def _resolve_color_enabled(self) -> bool:
        if self.color is not None:
            return self.color
        if self.printer is not print:
            return False
        return sys.stdout.isatty()

    def _reset_display_state_unlocked(self) -> None:
        self._pending_payer: str | None = None
        self._payer_run = 0
        self._last_summary_at: float | None = None
        self._last_model: str | None = None

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._snapshot = StatsSnapshot()
            self._reset_display_state_unlocked()

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
            self._last_summary_at = now
            self._last_model = None
            if todo_requests == 0:
                self._snapshot.eta_seconds = 0.0
                self._snapshot.estimated_finished_at = now
            else:
                self._snapshot.eta_seconds = None
                self._snapshot.estimated_finished_at = None
            snapshot = self._copy_snapshot_unlocked()

        if self.printer is not None and not self.verbose:
            self.printer(self.format_header(snapshot))
        return snapshot

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
            current_payer=self._snapshot.current_payer,
            payer_switches=self._snapshot.payer_switches,
            openai_requests=self._snapshot.openai_requests,
            developer_requests=self._snapshot.developer_requests,
            unknown_payer_requests=self._snapshot.unknown_payer_requests,
            by_model={model: ModelStats(**stats.to_dict()) for model, stats in self._snapshot.by_model.items()},
        )

    def _observed_payer(self, response: NormalizedResponse) -> str | None:
        if response.cost is not None:
            payer = response.cost.payer
        else:
            payer = response.billing.get("payer") if isinstance(response.billing, dict) else None
        return payer if payer in _KNOWN_PAYERS else None

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

            observed_payer = self._observed_payer(response)
            if observed_payer == "openai":
                self._snapshot.openai_requests += 1
            elif observed_payer == "developer":
                self._snapshot.developer_requests += 1
            else:
                self._snapshot.unknown_payer_requests += 1

            payer_change: tuple[str | None, str] | None = None
            if observed_payer is not None:
                if observed_payer == self._pending_payer:
                    self._payer_run += 1
                else:
                    self._pending_payer = observed_payer
                    self._payer_run = 1

                if (
                    self._payer_run >= self.payer_switch_threshold
                    and observed_payer != self._snapshot.current_payer
                ):
                    previous_payer = self._snapshot.current_payer
                    payer_change = (previous_payer, observed_payer)
                    if previous_payer in _KNOWN_PAYERS:
                        self._snapshot.payer_switches += 1
                    self._snapshot.current_payer = observed_payer

            self._snapshot.rolling_rpm = float(len(self._events))
            self._snapshot.rolling_tpm = float(sum(tokens for _, tokens in self._events))
            self._recompute_time_fields_unlocked(now)

            show_model = response.model != self._last_model
            self._last_model = response.model
            summary_due = self._snapshot.processed_requests % self.summary_every == 0
            if (
                self.summary_interval is not None
                and self._last_summary_at is not None
                and now - self._last_summary_at >= self.summary_interval
            ):
                summary_due = True
            if summary_due:
                self._last_summary_at = now
            snapshot = self._copy_snapshot_unlocked()

        if self.printer is not None:
            if self.verbose:
                self.printer(self.format_update(response, snapshot))
            else:
                if payer_change is not None:
                    self.printer(self.format_payer_change(*payer_change, snapshot))
                self.printer(self.format_request(response, snapshot, show_model=show_model))
                if summary_due:
                    self.printer(self.format_summary(snapshot))
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
            snapshot = self._copy_snapshot_unlocked()

        if self.printer is not None and not self.verbose:
            for line in self.format_final(snapshot):
                self.printer(line)
        return snapshot

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

    def _style(self, text: str, *codes: str) -> str:
        if not self._color_enabled:
            return text
        return "".join(codes) + text + _ANSI_RESET

    def format_header(self, snapshot: StatsSnapshot) -> str:
        if snapshot.skipped_requests == 0:
            return f"tokenrail · {snapshot.total_requests} requests"
        return (
            f"tokenrail · {snapshot.total_requests} requests "
            f"({snapshot.todo_requests} todo / {snapshot.skipped_requests} skipped)"
        )

    def format_request(
        self,
        response: NormalizedResponse,
        snapshot: StatsSnapshot,
        *,
        show_model: bool = False,
    ) -> str:
        sequence_width = max(4, len(str(max(snapshot.todo_requests, snapshot.processed_requests))))
        sequence = f"{snapshot.processed_requests:0{sequence_width}d}"
        status = "ok" if response.error is None else "ERR"
        prefix = f"  {sequence}  {status:<3}  {response.id:<12}"
        if show_model:
            prefix += f"  model={response.model}"
        if response.error is not None:
            return f"{prefix}  {response.error}"

        total_tokens = response.usage.total_tokens or (response.usage.input_tokens + response.usage.output_tokens)
        token_text = f"{_format_tokens(total_tokens)} tok"
        if response.usage.reasoning_tokens:
            token_text += f" (+{_format_tokens(response.usage.reasoning_tokens)} reasoning)"
        if response.usage.input_tokens > 0:
            cached_pct = round(response.usage.cached_tokens / response.usage.input_tokens * 100)
            token_text += f" ({cached_pct}% cached)"

        cost_text = _format_usd(response.cost.nominal_usd, digits=6) if response.cost is not None else "$—"
        payer = self._observed_payer(response)
        if payer == "developer":
            payer_tag = self._style("DEV", _ANSI_BOLD)
        elif payer == "openai":
            payer_tag = self._style("oai", _ANSI_DIM)
        else:
            payer_tag = self._style("?", _ANSI_DIM)

        fields = [prefix, token_text, cost_text, payer_tag]
        if response.timing is not None:
            fields.append(f"{response.timing.latency_seconds:.1f}s")
        return "  ".join(fields)

    def format_payer_change(self, previous: str | None, new: str, snapshot: StatsSnapshot) -> str:
        if previous is None:
            if new == "developer":
                return self._style("!! PAYER: developer — requests are billed to you", _ANSI_YELLOW, _ANSI_BOLD)
            return "   PAYER: openai — costs are covered"

        position = f"{snapshot.processed_requests}/{snapshot.todo_requests}"
        elapsed = self._format_duration(snapshot.elapsed_seconds)
        if new == "developer":
            line = f"!! PAYER SWITCH: openai → developer at {position} ({elapsed}) — now billed to you"
            return self._style(line, _ANSI_YELLOW, _ANSI_BOLD)
        return f"   PAYER SWITCH: developer → openai at {position} ({elapsed}) — costs are covered"

    def format_summary(self, snapshot: StatsSnapshot) -> str:
        pct = round(snapshot.processed_requests / snapshot.todo_requests * 100) if snapshot.todo_requests else 100
        eta = self._format_duration(snapshot.eta_seconds) if snapshot.eta_seconds is not None else "—"
        if snapshot.nominal_usd > 0:
            openai_pct = f"{round(snapshot.openai_usd / snapshot.nominal_usd * 100)}%"
        else:
            openai_pct = "—"
        return (
            f"── {snapshot.processed_requests}/{snapshot.todo_requests} · {pct}% · "
            f"{self._format_duration(snapshot.elapsed_seconds)} · ETA {eta} · "
            f"{snapshot.rolling_rpm:.0f} rpm · {_format_tokens(round(snapshot.rolling_tpm))} tpm · "
            f"{_format_usd(snapshot.nominal_usd, digits=3)} "
            f"(oai {openai_pct} / dev {_format_usd(snapshot.developer_usd, digits=3)})"
        )

    def format_final(self, snapshot: StatsSnapshot) -> list[str]:
        lines = [
            f"Done {snapshot.processed_requests}/{snapshot.todo_requests} · "
            f"{snapshot.success_requests} ok / {snapshot.error_requests} errors · "
            f"{self._format_duration(snapshot.elapsed_seconds)}"
        ]
        if snapshot.nominal_usd > 0:
            openai_pct = f"{round(snapshot.openai_usd / snapshot.nominal_usd * 100)}%"
            developer_pct = f"{round(snapshot.developer_usd / snapshot.nominal_usd * 100)}%"
        else:
            openai_pct = developer_pct = "—"
        lines.append(
            f"Total {_format_usd(snapshot.nominal_usd, digits=3)} — "
            f"openai {_format_usd(snapshot.openai_usd, digits=3)} ({openai_pct}) / "
            f"developer {_format_usd(snapshot.developer_usd, digits=3)} ({developer_pct})"
        )
        if snapshot.payer_switches:
            lines.append(f"Payer switches: {snapshot.payer_switches}")
        if len(snapshot.by_model) >= 2:
            for model, stats in sorted(snapshot.by_model.items()):
                lines.append(
                    f"  {model:<20} {stats.requests} req · {_format_tokens(stats.total_tokens)} tok · "
                    f"{_format_usd(stats.nominal_usd, digits=3)}"
                )
        return lines

    def format_update(self, response: NormalizedResponse, snapshot: StatsSnapshot) -> str:
        """Return the deprecated legacy progress line used by ``verbose=True``."""
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
