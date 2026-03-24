from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenrail.client import RailClient
from tokenrail.providers.openai import OpenAIProvider
from tokenrail.sinks import PerRequestJsonSink, ResultsJsonlSink


class _FakeResponsesAPI:
    def __init__(self, payloads, error=None):
        self.payloads = payloads
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            err = self.error.pop(0)
            if err is not None:
                raise err
        return self.payloads.pop(0)


class _FakeClient:
    def __init__(self, responses_api):
        self.responses = responses_api


class FakeRateLimitError(Exception):
    pass


class OpenAIProviderTests(unittest.TestCase):
    def test_reasoning_effort_is_blocked_for_gpt41(self):
        provider = OpenAIProvider(client=_FakeClient(_FakeResponsesAPI([])))
        with self.assertRaises(ValueError):
            provider.build_payload(model="gpt-4.1", input="hello", reasoning_effort="high")

    def test_supported_payload_fields_are_sent(self):
        payload = {
            "id": "resp_1",
            "model": "gpt-5.4-mini-2026-03-17",
            "service_tier": "default",
            "output_text": "done",
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 15,
            },
            "billing": {"payer": "developer"},
        }
        api = _FakeResponsesAPI([payload])
        provider = OpenAIProvider(client=_FakeClient(api))
        response = provider.create(
            model="gpt-5.4-mini-2026-03-17",
            input="hello",
            reasoning_effort="medium",
            verbosity="low",
            temperature=0.2,
            max_output_tokens=32,
        )
        sent = api.calls[0]
        self.assertEqual(sent["reasoning"]["effort"], "medium")
        self.assertEqual(sent["text"]["verbosity"], "low")
        self.assertEqual(sent["temperature"], 0.2)
        self.assertEqual(sent["max_output_tokens"], 32)
        self.assertEqual(response.usage.cached_tokens, 2)
        self.assertGreater(response.cost.developer_usd, 0.0)

    def test_retry_then_success(self):
        payload = {
            "id": "resp_2",
            "model": "gpt-5.4-mini-2026-03-17",
            "service_tier": "default",
            "output_text": "ok",
            "usage": {
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 3,
            },
            "billing": {"payer": "openai"},
        }
        api = _FakeResponsesAPI([payload], error=[FakeRateLimitError("retry"), None])
        provider = OpenAIProvider(
            client=_FakeClient(api),
            max_retries=2,
            base_sleep=0,
            retry_exceptions=(FakeRateLimitError,),
        )
        response = provider.create(model="gpt-5.4-mini-2026-03-17", input="hello", request_id="job-1")
        self.assertEqual(response.id, "job-1")
        self.assertEqual(len(api.calls), 2)
        self.assertAlmostEqual(response.cost.openai_usd, response.cost.nominal_usd)


class SinkTests(unittest.TestCase):
    def test_per_request_and_jsonl_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "records"
            jsonl_path = Path(tmp) / "results.jsonl"
            per_request = PerRequestJsonSink(output_dir)
            jsonl = ResultsJsonlSink(jsonl_path)
            client = RailClient.openai(
                client=_FakeClient(
                    _FakeResponsesAPI(
                        [
                            {
                                "id": "resp_3",
                                "model": "gpt-5.4-mini-2026-03-17",
                                "service_tier": "default",
                                "output_text": "saved",
                                "usage": {
                                    "input_tokens": 1,
                                    "input_tokens_details": {"cached_tokens": 0},
                                    "output_tokens": 1,
                                    "output_tokens_details": {"reasoning_tokens": 0},
                                    "total_tokens": 2,
                                },
                                "billing": {"payer": "developer"},
                            }
                        ]
                    )
                )
            )
            response = client.responses.create(model="gpt-5.4-mini-2026-03-17", input="hello", request_id="42")
            per_request.save(response)
            jsonl.save(response)
            self.assertEqual(per_request.load_done_ids(), {"42"})
            self.assertEqual(jsonl.load_done_ids(), {"42"})


if __name__ == "__main__":
    unittest.main()
