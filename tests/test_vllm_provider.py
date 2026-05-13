from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from tokenrail.client import RailClient
from tokenrail.executor import BatchExecutor, BatchItem
from tokenrail.monitor import RollingMetricsMonitor
from tokenrail.providers.vllm import VLLMProvider


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        parts = [f"{message['role']}::{message['content']}" for message in messages]
        if kwargs:
            parts.append(f"kwargs={kwargs}")
        return " | ".join(parts)


class _FakeCompletion:
    def __init__(self, text, token_ids, finish_reason="stop"):
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeGeneration:
    def __init__(self, prompt, text, prompt_token_ids, token_ids):
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [_FakeCompletion(text=text, token_ids=token_ids)]


class _FakeLLM:
    def __init__(self, tokenizer, responses_by_prompt):
        self._tokenizer = tokenizer
        self.responses_by_prompt = responses_by_prompt
        self.generate_calls = []

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.generate_calls.append(
            {
                "prompts": list(prompts),
                "sampling_params": sampling_params.kwargs,
                "use_tqdm": use_tqdm,
            }
        )
        return [self.responses_by_prompt[prompt] for prompt in prompts]


class _FakeRuntimeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self._tokenizer


def _fake_vllm_module() -> types.ModuleType:
    module = types.ModuleType("vllm")
    module.LLM = _FakeRuntimeLLM
    module.SamplingParams = _FakeSamplingParams
    return module


class VLLMProviderTests(unittest.TestCase):
    def test_railclient_vllm_constructs_provider_and_hf_is_removed(self):
        client = RailClient.vllm(model_id="Qwen/Qwen3.5-9B", family="qwen")
        self.assertIsInstance(client.provider, VLLMProvider)
        self.assertFalse(hasattr(RailClient, "hf"))

    def test_railclient_vllm_forwards_runtime_options(self):
        client = RailClient.vllm(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            device="cpu",
            metal_memory_fraction=0.7,
            extra_llm_kwargs={"max_num_seqs": 8},
        )

        self.assertEqual(client.provider.device, "cpu")
        self.assertEqual(client.provider.metal_memory_fraction, "0.7")
        self.assertEqual(client.provider.extra_llm_kwargs, {"max_num_seqs": 8})

    def test_runtime_options_are_forwarded_to_llm(self):
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            device="cpu",
            extra_llm_kwargs={"max_num_seqs": 8},
        )

        with patch.dict(sys.modules, {"vllm": _fake_vllm_module()}):
            llm, _, _ = provider._load_runtime()

        self.assertEqual(llm.kwargs["device"], "cpu")
        self.assertEqual(llm.kwargs["max_num_seqs"], 8)

    def test_extra_llm_kwargs_cannot_override_explicit_options(self):
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            extra_llm_kwargs={"dtype": "float16"},
        )

        with patch.dict(sys.modules, {"vllm": _fake_vllm_module()}):
            with self.assertRaisesRegex(ValueError, "dtype"):
                provider._load_runtime()

    def test_metal_memory_fraction_sets_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"vllm": _fake_vllm_module()}):
            VLLMProvider(
                model_id="Qwen/Qwen3.5-9B",
                family="qwen",
                metal_memory_fraction="auto",
            )._load_runtime()
            self.assertEqual(os.environ["VLLM_METAL_MEMORY_FRACTION"], "auto")

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"vllm": _fake_vllm_module()}):
            VLLMProvider(
                model_id="Qwen/Qwen3.5-9B",
                family="qwen",
                metal_memory_fraction=0.7,
            )._load_runtime()
            self.assertEqual(os.environ["VLLM_METAL_MEMORY_FRACTION"], "0.7")

    def test_invalid_metal_memory_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metal_memory_fraction"):
            VLLMProvider(model_id="Qwen/Qwen3.5-9B", family="qwen", metal_memory_fraction=1.5)
        with self.assertRaisesRegex(ValueError, "metal_memory_fraction"):
            VLLMProvider(model_id="Qwen/Qwen3.5-9B", family="qwen", metal_memory_fraction="fast")

    def test_macos_arm64_import_error_mentions_vllm_metal(self):
        provider = VLLMProvider(model_id="Qwen/Qwen3.5-9B", family="qwen")

        with patch("tokenrail.providers.vllm._is_macos_arm64", return_value=True):
            with patch.dict(sys.modules, {"vllm": None}):
                with self.assertRaises(ImportError) as raised:
                    provider._load_runtime()
        message = str(raised.exception)
        self.assertIn("vllm-metal", message)
        self.assertIn("Python 3.12", message)
        self.assertIn("tokenrail[vllm]", message)
        self.assertIn("raw.githubusercontent.com/vllm-project/vllm-metal", message)

    def test_qwen_prompt_passes_enable_thinking_to_chat_template(self):
        tokenizer = _FakeTokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "placeholder"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        llm = _FakeLLM(
            tokenizer,
            {
                prompt: _FakeGeneration(prompt, "answer", [1, 2], [3]),
                "user::hello | kwargs={'enable_thinking': True}": _FakeGeneration(
                    "user::hello | kwargs={'enable_thinking': True}",
                    "answer",
                    [1, 2],
                    [3],
                ),
            },
        )
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=_FakeSamplingParams,
        )

        response = provider.create(
            model="Qwen/Qwen3.5-9B",
            input="hello",
            enable_thinking=True,
        )

        self.assertEqual(response.output_text, "answer")
        self.assertTrue(tokenizer.calls[-1]["kwargs"]["enable_thinking"])

    def test_gemma_prompt_prefixes_system_prompt_when_thinking_enabled(self):
        tokenizer = _FakeTokenizer()
        prompt = "system::<|think|>policy | user::hello"
        llm = _FakeLLM(
            tokenizer,
            {
                prompt: _FakeGeneration(prompt, "answer", [1], [2, 3]),
            },
        )
        provider = VLLMProvider(
            model_id="google/gemma-4-e4b-it",
            family="gemma",
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=_FakeSamplingParams,
        )

        provider.create(
            model="google/gemma-4-e4b-it",
            input=[
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hello"},
            ],
            enable_thinking=True,
        )

        self.assertEqual(tokenizer.calls[-1]["messages"][0]["content"], "<|think|>policy")

    def test_create_many_groups_requests_by_sampling_configuration(self):
        tokenizer = _FakeTokenizer()
        prompt_a = "user::first | kwargs={'enable_thinking': False}"
        prompt_b = "user::second | kwargs={'enable_thinking': False}"
        prompt_c = "user::third | kwargs={'enable_thinking': True}"
        llm = _FakeLLM(
            tokenizer,
            {
                prompt_a: _FakeGeneration(prompt_a, "one", [1, 2, 3], [4]),
                prompt_b: _FakeGeneration(prompt_b, "two", [1, 2], [3, 4]),
                prompt_c: _FakeGeneration(prompt_c, "<think>hidden</think>three", [1], [2, 3, 4]),
            },
        )
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=_FakeSamplingParams,
        )

        responses = provider.create_many(
            [
                {"request_id": "a", "model": "Qwen/Qwen3.5-9B", "input": "first"},
                {"request_id": "b", "model": "Qwen/Qwen3.5-9B", "input": "second"},
                {"request_id": "c", "model": "Qwen/Qwen3.5-9B", "input": "third", "enable_thinking": True},
            ]
        )

        self.assertEqual(len(llm.generate_calls), 2)
        self.assertEqual(llm.generate_calls[0]["sampling_params"]["max_tokens"], 256)
        self.assertEqual(llm.generate_calls[1]["sampling_params"]["max_tokens"], 2048)
        self.assertEqual([response.id for response in responses], ["a", "b", "c"])
        self.assertEqual(responses[2].output_text, "three")

    def test_usage_and_unsupported_fields_are_validated(self):
        tokenizer = _FakeTokenizer()
        prompt = "user::hello | kwargs={'enable_thinking': False}"
        llm = _FakeLLM(
            tokenizer,
            {
                prompt: _FakeGeneration(prompt, "answer", [1, 2, 3], [4, 5]),
            },
        )
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=_FakeSamplingParams,
        )

        response = provider.create(model="Qwen/Qwen3.5-9B", input="hello")
        self.assertEqual(response.usage.input_tokens, 3)
        self.assertEqual(response.usage.output_tokens, 2)
        self.assertEqual(response.usage.total_tokens, 5)

        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            provider.create(model="Qwen/Qwen3.5-9B", input="hello", reasoning_effort="high")
        with self.assertRaisesRegex(ValueError, "model_id"):
            provider.create(model="Qwen/Qwen3.5-32B", input="hello")

    def test_batch_executor_uses_vllm_batched_provider(self):
        tokenizer = _FakeTokenizer()
        prompt_a = "user::alpha | kwargs={'enable_thinking': False}"
        prompt_b = "user::beta | kwargs={'enable_thinking': False}"
        llm = _FakeLLM(
            tokenizer,
            {
                prompt_a: _FakeGeneration(prompt_a, "A", [1], [2]),
                prompt_b: _FakeGeneration(prompt_b, "B", [1], [2]),
            },
        )
        client = RailClient(
            VLLMProvider(
                model_id="Qwen/Qwen3.5-9B",
                family="qwen",
                batch_flush_size=8,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params_cls=_FakeSamplingParams,
            )
        )
        executor = BatchExecutor(client=client, max_workers=4, monitor=RollingMetricsMonitor(printer=None))

        stats = executor.run(
            [
                BatchItem(id="alpha", request_kwargs={"model": "Qwen/Qwen3.5-9B", "input": "alpha"}),
                BatchItem(id="beta", request_kwargs={"model": "Qwen/Qwen3.5-9B", "input": "beta"}),
            ]
        )

        self.assertEqual(len(llm.generate_calls), 1)
        self.assertEqual(stats.success_requests, 2)
        self.assertEqual(stats.error_requests, 0)

    def test_vllm_flushes_large_groups_in_chunks(self):
        tokenizer = _FakeTokenizer()
        prompt_a = "user::alpha | kwargs={'enable_thinking': False}"
        prompt_b = "user::beta | kwargs={'enable_thinking': False}"
        prompt_c = "user::gamma | kwargs={'enable_thinking': False}"
        llm = _FakeLLM(
            tokenizer,
            {
                prompt_a: _FakeGeneration(prompt_a, "A", [1], [2]),
                prompt_b: _FakeGeneration(prompt_b, "B", [1], [2]),
                prompt_c: _FakeGeneration(prompt_c, "C", [1], [2]),
            },
        )
        provider = VLLMProvider(
            model_id="Qwen/Qwen3.5-9B",
            family="qwen",
            batch_flush_size=2,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=_FakeSamplingParams,
        )

        responses = provider.create_many(
            [
                {"request_id": "alpha", "model": "Qwen/Qwen3.5-9B", "input": "alpha"},
                {"request_id": "beta", "model": "Qwen/Qwen3.5-9B", "input": "beta"},
                {"request_id": "gamma", "model": "Qwen/Qwen3.5-9B", "input": "gamma"},
            ]
        )

        self.assertEqual(len(llm.generate_calls), 2)
        self.assertEqual([len(call["prompts"]) for call in llm.generate_calls], [2, 1])
        self.assertEqual([response.id for response in responses], ["alpha", "beta", "gamma"])


if __name__ == "__main__":
    unittest.main()
