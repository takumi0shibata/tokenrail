from __future__ import annotations

import tempfile
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tokenrail
import tokenrail.providers as providers
from tokenrail.catalog import (
    ModelCatalogFallbackWarning,
    calculate_cost,
    get_model_capabilities,
    get_model_pricing,
)
from tokenrail.client import RailClient
from tokenrail.providers.openai import OpenAIProvider
from tokenrail.sinks import PerRequestJsonSink, ResultsJsonlSink
from tokenrail.types import UsageBreakdown


class _FakeResponsesAPI:
    def __init__(self, payloads, error=None):
        self.payloads = payloads
        self.error = error
        self.calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            err = self.error.pop(0)
            if err is not None:
                raise err
        return self.payloads.pop(0)

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.payloads.pop(0)


class _FakeClient:
    def __init__(self, responses_api):
        self.responses = responses_api


class _BuiltClient:
    def __init__(self):
        self.responses = _FakeResponsesAPI([])


class PackageMetadataTests(unittest.TestCase):
    def test_version_matches_installed_metadata(self):
        from importlib.metadata import version

        self.assertEqual(tokenrail.__version__, version("tokenrail"))
        self.assertIn("__version__", tokenrail.__all__)


class OpenAIProviderTests(unittest.TestCase):
    def test_package_exports_openai_only_provider_symbols(self):
        self.assertIn("OpenAIProvider", tokenrail.__all__)
        self.assertNotIn("VLLMProvider", tokenrail.__all__)
        self.assertNotIn("VLLMServerProvider", tokenrail.__all__)
        self.assertFalse(hasattr(tokenrail, "VLLMProvider"))
        self.assertFalse(hasattr(tokenrail, "VLLMServerProvider"))
        self.assertEqual(providers.__all__, ["OpenAIProvider"])
        self.assertFalse(hasattr(RailClient, "vllm"))
        self.assertFalse(hasattr(RailClient, "vllm_server"))

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

    def test_prompt_cache_options_are_merged_into_extra_body_for_create(self):
        payload = {
            "id": "resp_cache",
            "model": "gpt-5.6",
            "output_text": "done",
            "usage": {
                "input_tokens": 1_500,
                "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 700},
                "output_tokens": 10,
                "total_tokens": 1_510,
            },
        }
        api = _FakeResponsesAPI([payload])
        provider = OpenAIProvider(client=_FakeClient(api))

        response = provider.create(
            model="gpt-5.6",
            input="hello",
            prompt_cache_key="cache:shard-0",
            prompt_cache_options={"mode": "explicit"},
            extra_body={"custom": True},
        )

        self.assertEqual(api.calls[0]["prompt_cache_key"], "cache:shard-0")
        self.assertEqual(
            api.calls[0]["extra_body"],
            {"custom": True, "prompt_cache_options": {"mode": "explicit"}},
        )
        self.assertEqual(response.usage.cache_write_tokens, 700)

    def test_prompt_cache_options_are_merged_for_parse_and_conflicts_are_rejected(self):
        class ParsedShape:
            pass

        payload = {
            "id": "resp_cache_parse",
            "model": "gpt-5.6",
            "output_text": "{}",
            "usage": {"input_tokens": 1_024, "output_tokens": 1, "total_tokens": 1_025},
        }
        api = _FakeResponsesAPI([payload])
        provider = OpenAIProvider(client=_FakeClient(api))

        provider.parse(
            model="gpt-5.6",
            input="hello",
            text_format=ParsedShape,
            prompt_cache_key="cache:shard-1",
            prompt_cache_options={"mode": "explicit"},
        )

        self.assertEqual(api.parse_calls[0]["prompt_cache_key"], "cache:shard-1")
        self.assertEqual(
            api.parse_calls[0]["extra_body"],
            {"prompt_cache_options": {"mode": "explicit"}},
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            provider.build_payload(
                model="gpt-5.6",
                input="hello",
                prompt_cache_options={"mode": "explicit"},
                extra_body={"prompt_cache_options": {"mode": "implicit"}},
            )

    def test_current_openai_sdk_preserves_explicit_cache_request_and_usage_fields(self):
        import json

        import httpx
        from openai import OpenAI

        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "resp_sdk_cache",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-5.6",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "usage": {
                        "input_tokens": 1_500,
                        "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 700},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 1_500,
                    },
                },
            )

        sdk_client = OpenAI(
            api_key="sk-test",
            base_url="https://example.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        provider = OpenAIProvider(client=sdk_client)
        response = provider.create(
            model="gpt-5.6",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "shared",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            ],
            prompt_cache_key="sdk:shard-0",
            prompt_cache_options={"mode": "explicit"},
        )

        self.assertEqual(captured["body"]["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual(
            captured["body"]["input"][0]["content"][0]["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )
        self.assertEqual(response.usage.cache_write_tokens, 700)

    def test_parse_uses_text_format_and_normalizes_parsed_output(self):
        class ParsedShape:
            pass

        payload = {
            "id": "resp_2",
            "model": "gpt-5.4-mini-2026-03-17",
            "service_tier": "default",
            "output_text": '{"answer":"done"}',
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"answer":"done"}',
                            "parsed": {"answer": "done"},
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            "billing": {"payer": "developer"},
        }
        api = _FakeResponsesAPI([payload])
        provider = OpenAIProvider(client=_FakeClient(api))

        response = provider.parse(
            model="gpt-5.4-mini-2026-03-17",
            input="hello",
            text_format=ParsedShape,
            reasoning_effort="medium",
            verbosity="low",
            request_id="job-2",
        )

        sent = api.parse_calls[0]
        self.assertIs(sent["text_format"], ParsedShape)
        self.assertEqual(sent["reasoning"]["effort"], "medium")
        self.assertEqual(sent["verbosity"], "low")
        self.assertEqual(response.id, "job-2")
        self.assertEqual(response.output_parsed, {"answer": "done"})
        self.assertIsNone(response.refusal)
        self.assertEqual(response.output_text, '{"answer":"done"}')

    def test_parse_rejects_response_format(self):
        provider = OpenAIProvider(client=_FakeClient(_FakeResponsesAPI([])))

        with self.assertRaisesRegex(ValueError, "response_format and text_format"):
            provider.build_parse_payload(
                model="gpt-5.4-mini-2026-03-17",
                input="hello",
                text_format=dict,
                response_format={"type": "json_schema"},
            )

        with self.assertRaisesRegex(ValueError, "text_format is required"):
            provider.build_parse_payload(model="gpt-5.4-mini-2026-03-17", input="hello", text_format=None)

    def test_create_rejects_text_format(self):
        provider = OpenAIProvider(client=_FakeClient(_FakeResponsesAPI([])))

        with self.assertRaisesRegex(ValueError, "text_format requires responses.parse"):
            provider.build_payload(model="gpt-5.4-mini-2026-03-17", input="hello", text_format=dict)

    def test_create_propagates_exceptions_without_local_retry(self):
        api = _FakeResponsesAPI([], error=[RuntimeError("boom")])
        provider = OpenAIProvider(client=_FakeClient(api))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            provider.create(model="gpt-5.4-mini-2026-03-17", input="hello", request_id="job-1")
        self.assertEqual(len(api.calls), 1)

    def test_model_pricing_matches_delimited_substrings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pricing = get_model_pricing("GEE-123456-2026-gpt-5.2")
            self.assertIsNotNone(pricing)
            self.assertEqual(str(pricing.input_per_million), "1.75")

            pricing = get_model_pricing("vendor/gpt-5.4-mini")
            self.assertIsNotNone(pricing)
            self.assertEqual(str(pricing.input_per_million), "0.750")

            pricing = get_model_pricing("abcgpt-5.2xyz")
            self.assertIsNone(pricing)

        self.assertEqual(len(caught), 3)
        self.assertTrue(all(item.category is ModelCatalogFallbackWarning for item in caught))

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
        with self.assertWarnsRegex(ModelCatalogFallbackWarning, "pricing from 'gpt-4o-mini'"):
            capabilities = get_model_capabilities("foo/gpt-4o-mini")
        self.assertTrue(capabilities.verbosity)
        self.assertFalse(capabilities.reasoning_effort)

    def test_gpt56_catalog_prices_capabilities_and_alias(self):
        expected = {
            "gpt-5.6-sol": ("5.00", "0.50", "30.00", "6.25"),
            "gpt-5.6": ("5.00", "0.50", "30.00", "6.25"),
            "gpt-5.6-terra": ("2.00", "0.20", "12.00", "2.50"),
            "gpt-5.6-luna": ("0.20", "0.02", "1.20", "0.25"),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for model, prices in expected.items():
                pricing = get_model_pricing(model)
                self.assertIsNotNone(pricing)
                self.assertEqual(
                    (
                        str(pricing.input_per_million),
                        str(pricing.cached_input_per_million),
                        str(pricing.output_per_million),
                        str(pricing.cache_write_input_per_million),
                    ),
                    prices,
                )
                capabilities = get_model_capabilities(model)
                self.assertTrue(capabilities.reasoning_effort)
                self.assertTrue(capabilities.verbosity)
        self.assertEqual(caught, [])

    def test_gpt56_cost_calculation_uses_base_prices(self):
        usage = UsageBreakdown(input_tokens=2_000_000, cached_tokens=1_000_000, output_tokens=1_000_000)
        expected_costs = {
            "gpt-5.6-sol": 35.5,
            "gpt-5.6-terra": 14.2,
            "gpt-5.6-luna": 1.42,
        }
        for model, expected in expected_costs.items():
            cost = calculate_cost(model, usage, payer="developer")
            self.assertIsNotNone(cost)
            self.assertAlmostEqual(cost.nominal_usd, expected)
            self.assertAlmostEqual(cost.developer_usd, expected)

    def test_official_snapshot_suffix_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pricing = get_model_pricing("gpt-5.6-terra-2026-08-03")
        self.assertIsNotNone(pricing)
        self.assertEqual(str(pricing.input_per_million), "2.00")
        self.assertEqual(caught, [])

    def test_gpt56_cost_calculation_separates_cache_reads_and_writes(self):
        usage = UsageBreakdown(
            input_tokens=1_000_000,
            cached_tokens=200_000,
            cache_write_tokens=300_000,
            output_tokens=100_000,
        )
        expected_costs = {
            "gpt-5.6-sol": 7.475,
            "gpt-5.6-terra": 2.99,
            "gpt-5.6-luna": 0.299,
        }
        for model, expected in expected_costs.items():
            cost = calculate_cost(model, usage, payer="developer")
            self.assertIsNotNone(cost)
            self.assertAlmostEqual(cost.nominal_usd, expected)

    def test_catalog_fallback_warning_is_emitted_once_per_model(self):
        model = "gpt-5.7-once-only"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(get_model_pricing, [model] * 20))
            get_model_capabilities(model)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn("Model 'gpt-5.7-once-only' is not explicitly registered", message)
        self.assertIn("capabilities from 'gpt-5'", message)
        self.assertIn("pricing from 'gpt-5'", message)

    def test_unknown_model_warning_explains_missing_pricing(self):
        model = "brand-new-model-without-catalog-entry"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            capabilities = get_model_capabilities(model)
            pricing = get_model_pricing(model)
        self.assertFalse(capabilities.reasoning_effort)
        self.assertIsNone(pricing)
        self.assertEqual(len(caught), 1)
        self.assertIn("default capabilities", str(caught[0].message))
        self.assertIn("cost=None", str(caught[0].message))


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
