from __future__ import annotations

import unittest

from tokenrail.client import RailClient
from tokenrail.providers.vllm_server import VLLMServerProvider


class _FakeChatCompletionsAPI:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.payloads.pop(0)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


class _BuiltClient(_FakeClient):
    def __init__(self):
        super().__init__(_FakeChatCompletionsAPI([]))


class VLLMServerProviderTests(unittest.TestCase):
    def test_railclient_constructs_vllm_server_provider(self):
        client = RailClient.vllm_server(client=_FakeClient(_FakeChatCompletionsAPI([])))
        self.assertIsInstance(client.provider, VLLMServerProvider)

    def test_base_url_is_forwarded_when_building_client(self):
        captured = {}

        class TestProvider(VLLMServerProvider):
            def _build_client(self, *, api_key, base_url, timeout, max_retries):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                captured["timeout"] = timeout
                captured["max_retries"] = max_retries
                return _BuiltClient()

        TestProvider(
            api_key="test-key",
            base_url="http://localhost:8001/v1",
            timeout=12.0,
            max_retries=4,
        )
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "http://localhost:8001/v1")
        self.assertEqual(captured["timeout"], 12.0)
        self.assertEqual(captured["max_retries"], 4)

    def test_create_uses_chat_completions_and_normalizes_response(self):
        api = _FakeChatCompletionsAPI(
            [
                {
                    "id": "chatcmpl-1",
                    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ]
        )
        provider = VLLMServerProvider(client=_FakeClient(api))

        response = provider.create(
            request_id="q1",
            model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            input="hello",
            max_output_tokens=8,
            temperature=0.2,
        )

        sent = api.calls[0]
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(sent["max_tokens"], 8)
        self.assertEqual(sent["temperature"], 0.2)
        self.assertEqual(response.id, "q1")
        self.assertEqual(response.provider, "vllm_server")
        self.assertEqual(response.output_text, "ok")
        self.assertEqual(response.usage.input_tokens, 3)
        self.assertEqual(response.usage.output_tokens, 1)
        self.assertEqual(response.usage.total_tokens, 4)

    def test_extra_body_carries_vllm_specific_options(self):
        api = _FakeChatCompletionsAPI(
            [
                {
                    "id": "chatcmpl-1",
                    "model": "model",
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {},
                }
            ]
        )
        provider = VLLMServerProvider(client=_FakeClient(api))

        provider.create(
            model="model",
            input="hello",
            enable_thinking=True,
            top_k=20,
            extra_body={"foo": "bar"},
        )

        self.assertEqual(
            api.calls[0]["extra_body"],
            {
                "foo": "bar",
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
            },
        )

    def test_unsupported_responses_fields_are_rejected(self):
        provider = VLLMServerProvider(client=_FakeClient(_FakeChatCompletionsAPI([])))
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            provider.create(model="model", input="hello", reasoning_effort="high")


if __name__ == "__main__":
    unittest.main()
