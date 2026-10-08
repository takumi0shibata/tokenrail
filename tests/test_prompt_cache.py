from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openai import OpenAI
from pydantic import BaseModel

from tokenrail import BatchExecutor, RailClient, ResultsJsonlSink, RollingMetricsMonitor, batch_items_from_queries


class Answer(BaseModel):
    answer: str


def _sdk_client(requests: list[dict]) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1_500,
                    "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 700},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1_503,
                },
            },
        )

    return OpenAI(
        api_key="sk-test",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _input(question: str) -> list[dict]:
    return [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Shared reference material",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"role": "user", "content": question},
    ]


class PromptCachePassthroughTests(unittest.TestCase):
    def test_create_and_parse_preserve_standard_cache_settings_on_the_wire(self):
        cases = [
            {"model": "gpt-6.1-sol", "input": "One-off prompt", "instructions": "Top-level instructions"},
            {"model": "gpt-6.1-sol", "input": "One-off prompt", "prompt_cache_options": {"mode": "explicit"}},
            {
                "model": "gpt-6.1-sol",
                "input": _input("Question"),
                "prompt_cache_key": "customer-1",
                "prompt_cache_options": {
                    "mode": "explicit",
                    "ttl": "30m",
                    "prewarm": False,
                    "comparison_response_id": "resp_previous",
                },
                "previous_response_id": "resp_previous",
            },
            {
                "model": "gpt-5.4",
                "input": "Earlier model",
                "prompt_cache_key": "existing-key",
                "prompt_cache_retention": "24h",
            },
        ]
        requests: list[dict] = []
        with _sdk_client(requests) as sdk:
            client = RailClient.openai(client=sdk)
            for method in ("create", "parse"):
                for case in cases:
                    with self.subTest(method=method, case=case):
                        kwargs = copy.deepcopy(case)
                        if method == "parse":
                            kwargs["text_format"] = Answer
                        original = copy.deepcopy(kwargs)
                        response = getattr(client.responses, method)(**kwargs)
                        body = requests[-1]
                        self.assertEqual(body["input"], case["input"])
                        for field in (
                            "instructions",
                            "prompt_cache_options",
                            "prompt_cache_key",
                            "prompt_cache_retention",
                        ):
                            if field in case:
                                self.assertEqual(body[field], case[field])
                            else:
                                self.assertNotIn(field, body)
                        if "previous_response_id" in case:
                            self.assertEqual(body["previous_response_id"], case["previous_response_id"])
                        self.assertEqual(kwargs, original)
                        self.assertEqual(response.usage.cached_tokens, 500)
                        self.assertEqual(response.usage.cache_write_tokens, 700)
        self.assertEqual(len(requests), 8)

    def test_batch_preserves_inputs_and_keys_without_prewarming_or_per_key_throttling(self):
        requests: list[dict] = []
        output: list[str] = []
        options = {"mode": "explicit"}
        items = batch_items_from_queries(
            {str(index): _input(f"Question {index}") for index in range(20)},
            model="gpt-6.1-sol",
            prompt_cache_key="existing-key",
            prompt_cache_options=options,
        )
        original = copy.deepcopy(items)
        with _sdk_client(requests) as sdk, tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "results.jsonl"
            result_file.write_text('{"id":"0"}\n', encoding="utf-8")
            monitor = RollingMetricsMonitor(printer=output.append, summary_every=1, summary_interval=None, color=False)
            executor = BatchExecutor(
                client=RailClient.openai(client=sdk),
                max_workers=1,
                max_rpm=120,
                sinks=[ResultsJsonlSink(result_file)],
                monitor=monitor,
            )
            now = [0.0]

            def sleep(seconds: float) -> None:
                now[0] += seconds

            executor._time_fn = lambda: now[0]
            executor._sleep_fn = sleep
            stats = executor.run(items)
            records = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(items, original)
        self.assertEqual(now[0], 0.0)
        self.assertEqual(len(requests), 19)
        for body, item in zip(requests, items[1:], strict=True):
            self.assertEqual(body["input"], item.request_kwargs["input"])
            self.assertEqual(body["prompt_cache_key"], "existing-key")
            self.assertEqual(body["prompt_cache_options"], options)
        self.assertEqual(stats.skipped_requests, 1)
        self.assertEqual(stats.success_requests, 19)
        self.assertEqual(stats.cached_tokens, 19 * 500)
        self.assertEqual(stats.cache_write_tokens, 19 * 700)
        self.assertEqual(stats.by_model["gpt-6.1-sol"].cache_write_tokens, 19 * 700)
        self.assertAlmostEqual(stats.nominal_usd, 19 * 0.00243)
        self.assertEqual(stats.to_dict()["cache_write_tokens"], 19 * 700)
        self.assertEqual(records[1]["usage"]["cache_write_tokens"], 700)
        self.assertIn("cache r33%/w47%", monitor.format_summary(stats))
        self.assertIn("Prompt cache: 9.5k read / 13.3k written", output)


if __name__ == "__main__":
    unittest.main()
