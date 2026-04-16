from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenrail.catalog import get_model_capabilities, get_model_pricing
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


class _BuiltClient:
    def __init__(self):
        self.responses = _FakeResponsesAPI([])


class OpenAIProviderTests(unittest.TestCase):
    def test_base_url_is_forwarded_when_building_client(self):
        captured = {}

        class TestProvider(OpenAIProvider):
            def _build_client(self, *, api_key, organization, timeout, base_url, max_retries):
                captured["api_key"] = api_key
                captured["organization"] = organization
                captured["timeout"] = timeout
                captured["base_url"] = base_url
                captured["max_retries"] = max_retries
                return _BuiltClient()

        test_provider = TestProvider(
            api_key="test-key",
            organization="test-org",
            timeout=12.0,
            base_url="https://example.test/v1",
            max_retries=4,
        )
        self.assertEqual(captured["base_url"], "https://example.test/v1")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["organization"], "test-org")
        self.assertEqual(captured["timeout"], 12.0)
        self.assertEqual(captured["max_retries"], 4)
        self.assertIsInstance(test_provider._client, _BuiltClient)

    def test_railclient_openai_accepts_base_url_with_injected_client(self):
        client = RailClient.openai(
            client=_FakeClient(_FakeResponsesAPI([])),
            base_url="https://example.test/v1",
        )
        self.assertIsInstance(client.provider, OpenAIProvider)

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

    def test_create_propagates_exceptions_without_local_retry(self):
        api = _FakeResponsesAPI([], error=[RuntimeError("boom")])
        provider = OpenAIProvider(client=_FakeClient(api))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            provider.create(model="gpt-5.4-mini-2026-03-17", input="hello", request_id="job-1")
        self.assertEqual(len(api.calls), 1)

    def test_model_pricing_matches_delimited_substrings(self):
        pricing = get_model_pricing("GEE-123456-2026-gpt-5.2")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "1.75")

        pricing = get_model_pricing("vendor/gpt-5.4-mini")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "0.750")

        pricing = get_model_pricing("abcgpt-5.2xyz")
        self.assertIsNone(pricing)

    def test_longest_match_wins_for_model_rules(self):
        pricing = get_model_pricing("gpt-5.4-mini-2026-03-17")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "0.750")

        pricing = get_model_pricing("gpt-5.4-2026-03-05")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "2.50")

        pricing = get_model_pricing("gpt-5.4-nano-2026-03-17")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "0.20")

    def test_capability_matching_uses_same_lookup(self):
        capabilities = get_model_capabilities("foo/gpt-4o-mini")
        self.assertTrue(capabilities.verbosity)
        self.assertFalse(capabilities.reasoning_effort)


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
