from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenrail.client import RailClient
from tokenrail.executor import BatchExecutor, BatchItem, batch_items_from_queries
from tokenrail.monitor import RollingMetricsMonitor
from tokenrail.sinks import ResultsJsonlSink
from tokenrail.types import CostBreakdown, NormalizedResponse, TimingBreakdown, UsageBreakdown


class _FakeResponsesNamespace:
    def __init__(self):
        self.calls = []

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


class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.responses = _FakeResponsesNamespace()

    def create(self, **kwargs):
        return self.responses.create(**kwargs)


class _FakeClient:
    def __init__(self):
        self.provider = _FakeProvider()
        self.responses = self.provider.responses


class MonitorAndExecutorTests(unittest.TestCase):
    def test_monitor_aggregates_usage_and_cost(self):
        monitor = RollingMetricsMonitor(printer=None)
        monitor.start(total_requests=2, todo_requests=2, skipped_requests=0)
        response = NormalizedResponse(
            id="1",
            model="gpt-5.4-mini",
            provider="openai",
            output_text="hi",
            raw_response={},
            usage=UsageBreakdown(input_tokens=10, cached_tokens=2, output_tokens=4, reasoning_tokens=1, total_tokens=14),
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
            executor = BatchExecutor(client=_FakeClient(), max_workers=4, sinks=[sink], monitor=RollingMetricsMonitor(printer=None))
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

    def test_batch_items_from_queries(self):
        items = batch_items_from_queries({"1": [{"role": "user", "content": "hello"}]}, model="gpt-5.4-mini")
        self.assertEqual(items[0].id, "1")
        self.assertEqual(items[0].request_kwargs["model"], "gpt-5.4-mini")


if __name__ == "__main__":
    unittest.main()
