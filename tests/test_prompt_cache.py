from __future__ import annotations

import copy
import threading
import unittest

from tokenrail import BatchExecutor, PromptCacheConfig
from tokenrail.monitor import RollingMetricsMonitor
from tokenrail.prompt_cache import build_prompt_cache_plan
from tokenrail.types import BatchItem, NormalizedResponse, TimingBreakdown, UsageBreakdown


def _block(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def _shared_items(*, model: str = "gpt-5.6") -> list[BatchItem]:
    return [
        BatchItem(
            id="a",
            request_kwargs={
                "model": model,
                "input": [
                    {
                        "role": "developer",
                        "content": [_block("shared instructions"), _block("first variable")],
                    }
                ],
            },
        ),
        BatchItem(
            id="b",
            request_kwargs={
                "model": model,
                "input": [
                    {
                        "role": "developer",
                        "content": [_block("shared instructions"), _block("second variable")],
                    }
                ],
            },
        ),
    ]


class PromptCachePlanningTests(unittest.TestCase):
    def test_usage_breakdown_keeps_existing_positional_argument_order(self):
        usage = UsageBreakdown(10, 2, 3, 4, 13)

        self.assertEqual(usage.output_tokens, 3)
        self.assertEqual(usage.reasoning_tokens, 4)
        self.assertEqual(usage.total_tokens, 13)
        self.assertEqual(usage.cache_write_tokens, 0)

    def test_marks_longest_common_content_block_without_mutating_originals(self):
        items = _shared_items()
        originals = copy.deepcopy(items)

        plan = build_prompt_cache_plan(
            items,
            config=PromptCacheConfig(base_key="evaluation", shards=2),
            max_rpm=None,
            provider_name="openai",
        )

        self.assertEqual(items, originals)
        self.assertEqual(plan.num_shards, 2)
        for item in plan.items:
            content = item.request_kwargs["input"][0]["content"]
            self.assertEqual(content[0]["prompt_cache_breakpoint"], {"mode": "explicit"})
            self.assertNotIn("prompt_cache_breakpoint", content[1])
            self.assertEqual(item.request_kwargs["prompt_cache_options"], {"mode": "explicit"})
            self.assertRegex(item.request_kwargs["prompt_cache_key"], r"^evaluation:shard-[01]$")

    def test_shared_instructions_are_moved_to_developer_message_for_string_inputs(self):
        items = [
            BatchItem(id="a", request_kwargs={"model": "gpt-5.6", "instructions": "shared", "input": "one"}),
            BatchItem(id="b", request_kwargs={"model": "gpt-5.6", "instructions": "shared", "input": "two"}),
        ]

        plan = build_prompt_cache_plan(
            items,
            config=PromptCacheConfig(shards=1),
            max_rpm=None,
            provider_name="openai",
        )

        for item in plan.items:
            self.assertNotIn("instructions", item.request_kwargs)
            messages = item.request_kwargs["input"]
            self.assertEqual(messages[0]["role"], "developer")
            self.assertEqual(messages[0]["content"][0]["text"], "shared")
            self.assertEqual(
                messages[0]["content"][0]["prompt_cache_breakpoint"],
                {"mode": "explicit"},
            )
            self.assertEqual(messages[1]["role"], "user")

    def test_auto_shard_count_uses_expected_or_executor_rpm(self):
        for rpm, expected in [(30, 2), (60, 4), (100, 7), (120, 8), (300, 20)]:
            with self.subTest(rpm=rpm):
                plan = build_prompt_cache_plan(
                    _shared_items(),
                    config=PromptCacheConfig(),
                    max_rpm=rpm,
                    provider_name="openai",
                )
                self.assertEqual(plan.num_shards, expected)

        override = build_prompt_cache_plan(
            _shared_items(),
            config=PromptCacheConfig(expected_rpm=45),
            max_rpm=300,
            provider_name="openai",
        )
        manual = build_prompt_cache_plan(
            _shared_items(),
            config=PromptCacheConfig(shards=3),
            max_rpm=300,
            provider_name="openai",
        )
        self.assertEqual(override.num_shards, 3)
        self.assertEqual(manual.num_shards, 3)

    def test_generated_base_key_and_assignments_are_stable_across_order(self):
        items = _shared_items()
        forward = build_prompt_cache_plan(
            items,
            config=PromptCacheConfig(shards=8),
            max_rpm=None,
            provider_name="openai",
        )
        reverse = build_prompt_cache_plan(
            list(reversed(items)),
            config=PromptCacheConfig(shards=8),
            max_rpm=None,
            provider_name="openai",
        )

        self.assertEqual(forward.base_key, reverse.base_key)
        forward_keys = {item.id: item.request_kwargs["prompt_cache_key"] for item in forward.items}
        reverse_keys = {item.id: item.request_kwargs["prompt_cache_key"] for item in reverse.items}
        self.assertEqual(forward_keys, reverse_keys)
        self.assertRegex(forward.base_key, r"^tokenrail:[0-9a-f]{24}$")

    def test_existing_prompt_cache_key_is_used_as_base(self):
        items = _shared_items()
        for item in items:
            item.request_kwargs["prompt_cache_key"] = "logical-key"

        plan = build_prompt_cache_plan(
            items,
            config=PromptCacheConfig(shards=2),
            max_rpm=None,
            provider_name="openai",
        )

        self.assertEqual(plan.base_key, "logical-key")
        self.assertTrue(
            all(item.request_kwargs["prompt_cache_key"].startswith("logical-key:shard-") for item in plan.items)
        )

    def test_invalid_batches_fail_before_planning(self):
        invalid_cases: list[tuple[str, list[BatchItem], str]] = [
            (
                "no common prefix",
                [
                    BatchItem(id="a", request_kwargs={"model": "gpt-5.6", "input": "one"}),
                    BatchItem(id="b", request_kwargs={"model": "gpt-5.6", "input": "two"}),
                ],
                "no common cacheable",
            ),
            ("unsupported model", _shared_items(model="gpt-5.5"), "not supported"),
            (
                "mixed config",
                [
                    BatchItem(
                        id="a",
                        request_kwargs={"model": "gpt-5.6", "instructions": "shared", "input": "a"},
                    ),
                    BatchItem(
                        id="b",
                        request_kwargs={
                            "model": "gpt-5.6",
                            "instructions": "shared",
                            "input": "b",
                            "temperature": 0.5,
                        },
                    ),
                ],
                "homogeneous",
            ),
            (
                "stateful",
                [
                    BatchItem(
                        id="a",
                        request_kwargs={
                            "model": "gpt-5.6",
                            "instructions": "shared",
                            "input": "a",
                            "previous_response_id": "resp_1",
                        },
                    )
                ],
                "stateful",
            ),
            (
                "existing options",
                [
                    BatchItem(
                        id="a",
                        request_kwargs={
                            "model": "gpt-5.6",
                            "instructions": "shared",
                            "input": "a",
                            "prompt_cache_options": {"mode": "explicit"},
                        },
                    )
                ],
                "conflicts",
            ),
        ]

        for name, items, message in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                build_prompt_cache_plan(
                    items,
                    config=PromptCacheConfig(shards=1),
                    max_rpm=None,
                    provider_name="openai",
                )

    def test_auto_shards_require_an_rpm_source(self):
        with self.assertRaisesRegex(ValueError, "requires max_rpm"):
            build_prompt_cache_plan(
                _shared_items(),
                config=PromptCacheConfig(),
                max_rpm=None,
                provider_name="openai",
            )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _TimedResponses:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, float, str]] = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append((kwargs["request_id"], self.clock.time(), kwargs["prompt_cache_key"]))
        return NormalizedResponse(
            id=kwargs["request_id"],
            model=kwargs["model"],
            provider="openai",
            output_text="ok",
            raw_response={},
            usage=UsageBreakdown(input_tokens=100, cache_write_tokens=50, output_tokens=1, total_tokens=101),
            timing=TimingBreakdown(started_at=0.0, completed_at=0.0, latency_seconds=0.0),
        )


class _OpenAIProvider:
    name = "openai"

    def __init__(self, clock: _Clock) -> None:
        self.responses = _TimedResponses(clock)


class _OpenAIClient:
    def __init__(self, clock: _Clock) -> None:
        self.provider = _OpenAIProvider(clock)
        self.responses = self.provider.responses


class PromptCacheExecutorTests(unittest.TestCase):
    def test_per_shard_limiter_and_stats_are_applied(self):
        clock = _Clock()
        client = _OpenAIClient(clock)
        executor = BatchExecutor(
            client=client,
            max_workers=4,
            prompt_cache=PromptCacheConfig(base_key="one", shards=1, target_rpm_per_shard=2),
            monitor=RollingMetricsMonitor(printer=None),
        )
        executor._time_fn = clock.time
        executor._sleep_fn = clock.sleep
        items = [
            BatchItem(
                id=str(index),
                request_kwargs={"model": "gpt-5.6", "instructions": "shared", "input": f"input {index}"},
            )
            for index in range(3)
        ]

        snapshot = executor.run(items)

        call_times = sorted(call_time for _, call_time, _ in client.responses.calls)
        self.assertEqual(call_times, [0.0, 0.0, 60.0])
        self.assertEqual(clock.sleeps, [60.0])
        self.assertEqual(snapshot.prompt_cache_shards, 1)
        self.assertEqual(snapshot.prompt_cache_target_rpm_per_shard, 2)
        self.assertEqual(snapshot.cache_write_tokens, 150)
        self.assertTrue(all(key == "one:shard-0" for _, _, key in client.responses.calls))

    def test_invalid_cache_plan_sends_no_requests(self):
        clock = _Clock()
        client = _OpenAIClient(clock)
        executor = BatchExecutor(
            client=client,
            max_rpm=30,
            prompt_cache="auto",
            monitor=RollingMetricsMonitor(printer=None),
        )

        with self.assertRaisesRegex(ValueError, "no common cacheable"):
            executor.run(
                [
                    BatchItem(id="a", request_kwargs={"model": "gpt-5.6", "input": "one"}),
                    BatchItem(id="b", request_kwargs={"model": "gpt-5.6", "input": "two"}),
                ]
            )

        self.assertEqual(client.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
