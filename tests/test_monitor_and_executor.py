from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tokenrail.executor import BatchExecutor, BatchItem, batch_items_from_queries
from tokenrail.monitor import RollingMetricsMonitor
from tokenrail.sinks import ResultsJsonlSink
from tokenrail.types import CostBreakdown, NormalizedResponse, TimingBreakdown, UsageBreakdown

_UNSET = object()


def _response(
    item_id: str,
    *,
    payer: str | None = "openai",
    model: str = "gpt-5.6-terra",
    with_cost: bool = True,
    billing_payer: str | None | object = _UNSET,
    error: str | None = None,
) -> NormalizedResponse:
    if billing_payer is _UNSET:
        billing_payer = payer
    billing = {"payer": billing_payer} if billing_payer is not None else None
    if with_cost:
        cost = CostBreakdown(
            nominal_usd=0.1,
            developer_usd=0.1 if payer != "openai" else 0.0,
            openai_usd=0.1 if payer == "openai" else 0.0,
            payer=payer,
        )
    else:
        cost = None
    usage = (
        UsageBreakdown.empty()
        if error is not None
        else UsageBreakdown(
            input_tokens=1_000,
            cached_tokens=400,
            output_tokens=280,
            reasoning_tokens=20,
            total_tokens=1_280,
        )
    )
    return NormalizedResponse(
        id=item_id,
        model=model,
        provider="openai",
        output_text=None if error else "ok",
        raw_response={},
        usage=usage,
        billing=billing,
        cost=cost,
        timing=TimingBreakdown(started_at=0.0, completed_at=1.4, latency_seconds=1.4),
        error=error,
    )


class _FakeResponsesNamespace:
    def __init__(self):
        self.calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["input"] == "boom":
            raise RuntimeError("failed")
        usage = UsageBreakdown(
            input_tokens=4,
            cached_tokens=1,
            output_tokens=2,
            reasoning_tokens=1,
            total_tokens=6,
        )
        return NormalizedResponse(
            id=kwargs["request_id"],
            model=kwargs["model"],
            provider="fake",
            output_text=f"ok:{kwargs['input']}",
            raw_response={"id": kwargs["request_id"]},
            usage=usage,
            billing={"payer": "developer"},
            cost=CostBreakdown(nominal_usd=0.1, developer_usd=0.1, openai_usd=0.0, payer="developer"),
            timing=TimingBreakdown(started_at=1.0, completed_at=2.0, latency_seconds=1.0),
        )

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        usage = UsageBreakdown(total_tokens=6)
        return NormalizedResponse(
            id=kwargs["request_id"],
            model=kwargs["model"],
            provider="fake",
            output_text=f"ok:{kwargs['input']}",
            output_parsed={"parsed": kwargs["input"]},
            raw_response={"id": kwargs["request_id"]},
            usage=usage,
            timing=TimingBreakdown(started_at=1.0, completed_at=2.0, latency_seconds=1.0),
        )


class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.responses = _FakeResponsesNamespace()

    def create(self, **kwargs):
        return self.responses.create(**kwargs)

    def parse(self, **kwargs):
        return self.responses.parse(**kwargs)


class _FakeClient:
    def __init__(self):
        self.provider = _FakeProvider()
        self.responses = self.provider.responses


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _TimedResponsesNamespace:
    def __init__(self, clock: _Clock, total_tokens: int = 6):
        self.clock = clock
        self.total_tokens = total_tokens
        self.calls = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append((kwargs["request_id"], self.clock.time()))
        return NormalizedResponse(
            id=kwargs["request_id"],
            model=kwargs["model"],
            provider="fake",
            output_text=f"ok:{kwargs['input']}",
            raw_response={"id": kwargs["request_id"]},
            usage=UsageBreakdown(total_tokens=self.total_tokens),
            timing=TimingBreakdown(started_at=0.0, completed_at=0.0, latency_seconds=0.0),
        )


class _TimedProvider:
    name = "fake"

    def __init__(self, clock: _Clock, total_tokens: int = 6):
        self.responses = _TimedResponsesNamespace(clock, total_tokens=total_tokens)


class _TimedClient:
    def __init__(self, clock: _Clock, total_tokens: int = 6):
        self.provider = _TimedProvider(clock, total_tokens=total_tokens)
        self.responses = self.provider.responses


class MonitorAndExecutorTests(unittest.TestCase):
    def test_monitor_constructor_validates_new_options(self):
        with self.assertRaisesRegex(ValueError, "summary_every"):
            RollingMetricsMonitor(summary_every=0)
        with self.assertRaisesRegex(ValueError, "summary_interval"):
            RollingMetricsMonitor(summary_interval=0)
        with self.assertRaisesRegex(ValueError, "payer_switch_threshold"):
            RollingMetricsMonitor(payer_switch_threshold=0)

    def test_default_output_separates_header_payer_and_request_details(self):
        output = []
        monitor = RollingMetricsMonitor(
            printer=output.append,
            summary_interval=None,
            payer_switch_threshold=1,
            color=False,
        )
        monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
        monitor.record(_response("req-0001"))
        monitor.record(_response("req-0002"))

        self.assertEqual(output[0], "tokenrail · 2 requests")
        self.assertEqual(output[1], "   PAYER: openai — costs are covered")
        self.assertIn("0001  ok", output[2])
        self.assertIn("model=gpt-5.6-terra", output[2])
        self.assertIn("1.3k tok (+20 reasoning) (40% cached)", output[2])
        self.assertIn("$0.100000", output[2])
        self.assertIn("oai", output[2])
        self.assertIn("1.4s", output[2])
        self.assertNotIn("model=", output[3])

    def test_model_name_is_repeated_only_when_it_changes(self):
        output = []
        monitor = RollingMetricsMonitor(printer=output.append, summary_interval=None, color=False)
        monitor.start(total_requests=3, todo_requests=3, skipped_requests=0)
        monitor.record(_response("1", model="gpt-5.6-terra"))
        monitor.record(_response("2", model="gpt-5.6-terra"))
        monitor.record(_response("3", model="gpt-5.6-luna"))
        request_lines = [line for line in output if "  ok " in line]
        self.assertIn("model=gpt-5.6-terra", request_lines[0])
        self.assertNotIn("model=", request_lines[1])
        self.assertIn("model=gpt-5.6-luna", request_lines[2])

    def test_payer_hysteresis_tracks_initial_switch_and_recovery(self):
        output = []
        monitor = RollingMetricsMonitor(printer=output.append, summary_interval=None, color=False)
        monitor.start(total_requests=15, todo_requests=15, skipped_requests=0)
        for index in range(5):
            monitor.record(_response(f"oai-{index}", payer="openai"))
        for index in range(5):
            monitor.record(_response(f"dev-{index}", payer="developer"))
        for index in range(5):
            snapshot = monitor.record(_response(f"oai-again-{index}", payer="openai"))

        payer_lines = [line for line in output if "PAYER" in line]
        self.assertEqual(len(payer_lines), 3)
        self.assertIn("PAYER: openai", payer_lines[0])
        self.assertNotIn("SWITCH", payer_lines[0])
        self.assertTrue(payer_lines[1].startswith("!! PAYER SWITCH: openai → developer"))
        self.assertTrue(payer_lines[2].startswith("   PAYER SWITCH: developer → openai"))
        self.assertEqual(snapshot.current_payer, "openai")
        self.assertEqual(snapshot.payer_switches, 2)
        self.assertEqual(snapshot.openai_requests, 10)
        self.assertEqual(snapshot.developer_requests, 5)

    def test_unknown_payer_is_counted_but_does_not_drive_transitions(self):
        output = []
        monitor = RollingMetricsMonitor(printer=output.append, summary_interval=None, color=False)
        monitor.start(total_requests=8, todo_requests=8, skipped_requests=0)
        for index in range(3):
            monitor.record(_response(f"oai-{index}", payer="openai"))
        monitor.record(_response("dev-one", payer="developer"))
        monitor.record(_response("unknown", payer=None, with_cost=False, billing_payer=None))
        monitor.record(_response("oai-again", payer="openai"))
        snapshot = monitor.snapshot()

        self.assertEqual(snapshot.current_payer, "openai")
        self.assertEqual(snapshot.payer_switches, 0)
        self.assertEqual(snapshot.developer_requests, 1)
        self.assertEqual(snapshot.unknown_payer_requests, 1)
        self.assertFalse(any("PAYER SWITCH" in line for line in output))

    def test_billing_payer_is_used_when_price_is_unavailable(self):
        output = []
        monitor = RollingMetricsMonitor(
            printer=output.append,
            summary_interval=None,
            payer_switch_threshold=1,
            color=False,
        )
        monitor.start(total_requests=1, todo_requests=1, skipped_requests=0)
        snapshot = monitor.record(_response("unknown-price", with_cost=False, billing_payer="developer"))

        self.assertEqual(snapshot.current_payer, "developer")
        self.assertEqual(snapshot.developer_requests, 1)
        self.assertIn("!! PAYER: developer", output[1])
        self.assertIn("$—", output[2])
        self.assertIn("DEV", output[2])

    def test_summary_prints_by_count_and_time_without_duplicates(self):
        output = []
        monitor = RollingMetricsMonitor(
            printer=output.append,
            summary_every=2,
            summary_interval=None,
            color=False,
        )
        monitor.start(total_requests=4, todo_requests=4, skipped_requests=0)
        for index in range(4):
            monitor.record(_response(str(index)))
        summaries = [line for line in output if line.startswith("──")]
        self.assertEqual(len(summaries), 2)
        self.assertIn("2/4", summaries[0])
        self.assertIn("4/4", summaries[1])
        self.assertIn("oai 100%", summaries[1])

        timed_output = []
        timed_monitor = RollingMetricsMonitor(
            printer=timed_output.append,
            summary_every=50,
            summary_interval=30.0,
            color=False,
        )
        with patch("tokenrail.monitor.time.time", side_effect=[0.0, 10.0, 31.0]):
            timed_monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
            timed_monitor.record(_response("1"))
            timed_monitor.record(_response("2"))
        self.assertEqual(len([line for line in timed_output if line.startswith("──")]), 1)

    def test_finalize_prints_totals_and_sorted_model_breakdown(self):
        output = []
        monitor = RollingMetricsMonitor(printer=output.append, summary_interval=None, color=False)
        monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
        monitor.record(_response("b", model="gpt-5.6-terra", payer="developer"))
        monitor.record(_response("a", model="gpt-5.6-luna", payer="openai"))
        snapshot = monitor.finalize(total_requests=2, skipped_requests=0)

        final_lines = monitor.format_final(snapshot)
        self.assertTrue(final_lines[0].startswith("Done 2/2 · 2 ok / 0 errors"))
        self.assertIn("Total $0.200", final_lines[1])
        self.assertFalse(any(line.startswith("Payer switches:") for line in final_lines))
        model_lines = [line.strip() for line in final_lines if " req · " in line]
        self.assertTrue(model_lines[0].startswith("gpt-5.6-luna"))
        self.assertTrue(model_lines[1].startswith("gpt-5.6-terra"))
        self.assertEqual(output[-len(final_lines) :], final_lines)

    def test_new_snapshot_fields_survive_copy_and_serialization(self):
        monitor = RollingMetricsMonitor(printer=None, payer_switch_threshold=1)
        monitor.start(total_requests=1, todo_requests=1, skipped_requests=0)
        monitor.record(_response("1", payer="developer"))
        snapshot = monitor.snapshot()
        serialized = snapshot.to_dict()
        self.assertEqual(snapshot.current_payer, "developer")
        self.assertEqual(snapshot.developer_requests, 1)
        for key in (
            "current_payer",
            "payer_switches",
            "openai_requests",
            "developer_requests",
            "unknown_payer_requests",
        ):
            self.assertIn(key, serialized)

    def test_error_request_line_omits_token_and_cost_fields(self):
        monitor = RollingMetricsMonitor(printer=None, color=False)
        monitor.start(total_requests=1, todo_requests=1, skipped_requests=0)
        response = _response("failed", with_cost=False, billing_payer=None, error="RateLimitError: slow down")
        snapshot = monitor.record(response)
        line = monitor.format_request(response, snapshot, show_model=True)
        self.assertIn("ERR", line)
        self.assertIn("RateLimitError: slow down", line)
        self.assertNotIn("tok", line)
        self.assertNotIn("$", line)

    def test_verbose_mode_uses_only_legacy_request_lines(self):
        output = []
        monitor = RollingMetricsMonitor(printer=output.append, verbose=True, summary_every=1)
        monitor.start(total_requests=1, todo_requests=1, skipped_requests=0)
        monitor.record(_response("1"))
        monitor.finalize(total_requests=1, skipped_requests=0)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].startswith("[1/1] id=1 model=gpt-5.6-terra"))

    def test_forced_color_adds_ansi_without_changing_plain_text_mode(self):
        colored_output = []
        colored = RollingMetricsMonitor(
            printer=colored_output.append,
            summary_interval=None,
            payer_switch_threshold=1,
            color=True,
        )
        colored.start(total_requests=1, todo_requests=1, skipped_requests=0)
        colored.record(_response("1", payer="developer"))
        self.assertIn("\033[", colored_output[1])
        self.assertIn("\033[", colored_output[2])

        plain_output = []
        plain = RollingMetricsMonitor(
            printer=plain_output.append,
            summary_interval=None,
            payer_switch_threshold=1,
            color=False,
        )
        plain.start(total_requests=1, todo_requests=1, skipped_requests=0)
        plain.record(_response("1", payer="developer"))
        self.assertFalse(any("\033[" in line for line in plain_output))

    def test_monitor_aggregates_usage_and_cost(self):
        monitor = RollingMetricsMonitor(printer=None)
        monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
        response = NormalizedResponse(
            id="1",
            model="gpt-5.4-mini",
            provider="openai",
            output_text="hi",
            raw_response={},
            usage=UsageBreakdown(
                input_tokens=10, cached_tokens=2, output_tokens=4, reasoning_tokens=1, total_tokens=14
            ),
            billing={"payer": "openai"},
            cost=CostBreakdown(nominal_usd=0.5, developer_usd=0.0, openai_usd=0.5, payer="openai"),
            timing=TimingBreakdown(started_at=0.0, completed_at=1.0, latency_seconds=1.0),
        )
        snapshot = monitor.record(response)
        self.assertEqual(snapshot.success_requests, 1)
        self.assertEqual(snapshot.cached_tokens, 2)
        self.assertEqual(snapshot.rolling_rpm, 1.0)
        self.assertEqual(snapshot.rolling_tpm, 14.0)
        self.assertAlmostEqual(snapshot.openai_usd, 0.5)
        self.assertIsNotNone(snapshot.started_at)
        self.assertIsNotNone(snapshot.last_updated_at)
        self.assertGreaterEqual(snapshot.elapsed_seconds, 0.0)
        self.assertEqual(snapshot.remaining_requests, 1)
        self.assertIsNotNone(snapshot.eta_seconds)
        self.assertIsNotNone(snapshot.estimated_finished_at)

    def test_monitor_start_and_zero_todo_snapshot(self):
        monitor = RollingMetricsMonitor(printer=None)
        snapshot = monitor.start(total_requests=3, todo_requests=0, skipped_requests=3)
        self.assertEqual(snapshot.total_requests, 3)
        self.assertEqual(snapshot.todo_requests, 0)
        self.assertEqual(snapshot.skipped_requests, 3)
        self.assertEqual(snapshot.remaining_requests, 0)
        self.assertEqual(snapshot.elapsed_seconds, 0.0)
        self.assertEqual(snapshot.eta_seconds, 0.0)
        self.assertIsNotNone(snapshot.estimated_finished_at)

    def test_format_update_includes_elapsed_eta_and_finish(self):
        monitor = RollingMetricsMonitor(printer=None)
        monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
        response = NormalizedResponse(
            id="1",
            model="gpt-5.4-mini",
            provider="openai",
            output_text="hi",
            raw_response={},
            usage=UsageBreakdown(input_tokens=1, cached_tokens=0, output_tokens=1, reasoning_tokens=0, total_tokens=2),
            cost=CostBreakdown(nominal_usd=0.1, developer_usd=0.1, openai_usd=0.0, payer="developer"),
            timing=TimingBreakdown(started_at=0.0, completed_at=1.0, latency_seconds=1.0),
        )
        snapshot = monitor.record(response)
        line = monitor.format_update(response, snapshot)
        self.assertIn("elapsed=", line)
        self.assertIn("eta=", line)
        self.assertIn("finish=", line)
        self.assertIn("[1/2]", line)

    def test_results_jsonl_sink_is_thread_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = ResultsJsonlSink(Path(tmp) / "results.jsonl")
            responses = [
                NormalizedResponse(
                    id=str(i),
                    model="gpt-5.4-mini",
                    provider="openai",
                    output_text="x",
                    raw_response={"id": i},
                    usage=UsageBreakdown(total_tokens=1),
                )
                for i in range(50)
            ]
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(sink.save, responses))
            self.assertEqual(len(sink.load_done_ids()), 50)

    def test_batch_executor_handles_success_failure_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = ResultsJsonlSink(Path(tmp) / "results.jsonl")
            executor = BatchExecutor(
                client=_FakeClient(), max_workers=4, sinks=[sink], monitor=RollingMetricsMonitor(printer=None)
            )
            items = [
                BatchItem(id="a", request_kwargs={"model": "gpt-5.4-mini", "input": "ok"}),
                BatchItem(id="b", request_kwargs={"model": "gpt-5.4-mini", "input": "boom"}),
            ]
            first = executor.run(items)
            self.assertEqual(first.total_requests, 2)
            self.assertEqual(first.todo_requests, 2)
            self.assertEqual(first.success_requests, 1)
            self.assertEqual(first.error_requests, 1)
            self.assertEqual(first.remaining_requests, 0)
            self.assertEqual(first.eta_seconds, 0.0)
            self.assertIsNotNone(first.started_at)
            self.assertIsNotNone(first.last_updated_at)

            second = executor.run(items)
            self.assertEqual(second.total_requests, 2)
            self.assertEqual(second.todo_requests, 0)
            self.assertEqual(second.skipped_requests, 2)
            self.assertEqual(second.remaining_requests, 0)
            self.assertEqual(second.eta_seconds, 0.0)
            self.assertIsNotNone(second.estimated_finished_at)

    def test_batch_executor_uses_parse_when_text_format_is_present(self):
        class ParsedShape:
            pass

        client = _FakeClient()
        executor = BatchExecutor(client=client, max_workers=1, monitor=RollingMetricsMonitor(printer=None))
        items = [
            BatchItem(
                id="a",
                request_kwargs={"model": "gpt-5.4-mini", "input": "ok", "text_format": ParsedShape},
            )
        ]

        snapshot = executor.run(items)

        self.assertEqual(snapshot.success_requests, 1)
        self.assertEqual(len(client.responses.calls), 0)
        self.assertEqual(len(client.responses.parse_calls), 1)
        self.assertIs(client.responses.parse_calls[0]["text_format"], ParsedShape)
        self.assertEqual(client.responses.parse_calls[0]["request_id"], "a")

    def test_batch_executor_limits_submits_by_rpm(self):
        clock = _Clock()
        client = _TimedClient(clock)
        executor = BatchExecutor(
            client=client,
            max_workers=4,
            max_rpm=2,
            monitor=RollingMetricsMonitor(printer=None),
        )
        executor._time_fn = clock.time
        executor._sleep_fn = clock.sleep
        items = [
            BatchItem(id=str(i), request_kwargs={"model": "gpt-5.4-mini", "input": str(i)})
            for i in range(4)
        ]

        executor.run(items)

        call_times = sorted(call_time for _, call_time in client.responses.calls)
        self.assertEqual(call_times[:2], [0.0, 0.0])
        self.assertEqual(call_times[2:], [60.0, 60.0])
        self.assertEqual(clock.sleeps, [60.0])

    def test_batch_executor_limits_submits_by_tpm_estimate(self):
        clock = _Clock()
        client = _TimedClient(clock, total_tokens=6)
        executor = BatchExecutor(
            client=client,
            max_workers=4,
            max_tpm=10,
            monitor=RollingMetricsMonitor(printer=None),
        )
        executor._time_fn = clock.time
        executor._sleep_fn = clock.sleep
        items = [
            BatchItem(id=str(i), request_kwargs={"model": "gpt-5.4-mini", "input": str(i)})
            for i in range(3)
        ]

        executor.run(items)

        self.assertEqual([call_time for _, call_time in client.responses.calls], [0.0, 60.0, 120.0])
        self.assertEqual(clock.sleeps, [60.0, 60.0])

    def test_batch_items_from_queries(self):
        items = batch_items_from_queries({"1": [{"role": "user", "content": "hello"}]}, model="gpt-5.4-mini")
        self.assertEqual(items[0].id, "1")
        self.assertEqual(items[0].request_kwargs["model"], "gpt-5.4-mini")


if __name__ == "__main__":
    unittest.main()
