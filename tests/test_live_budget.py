from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from _live_budget import MODEL, TOKEN_BUDGET, LiveBudgetTransport
from openai import APIConnectionError, OpenAI
from pydantic import BaseModel

from tokenrail import RailClient


def raw_response(**overrides):
    return {
        "id": "resp_budget", "object": "response", "created_at": 0,
        "model": MODEL, "status": "completed", "output": [],
        "billing": {"payer": "openai"},
        "usage": {
            "input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        **overrides,
    }


def setup(*, remaining=None, response=None, count=10, retries=0, generation_error=False):
    calls = []

    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("input_tokens"):
            value = count() if callable(count) else count
            return httpx.Response(200, json={"input_tokens": value})
        if generation_error:
            raise httpx.ReadTimeout("simulated timeout")
        raw = response() if callable(response) else response
        return httpx.Response(200, json=raw if raw is not None else raw_response())

    guard = LiveBudgetTransport(httpx.MockTransport(handler), free_tokens_remaining=remaining)
    sdk = OpenAI(api_key="test-placeholder", max_retries=retries, http_client=httpx.Client(transport=guard))
    return sdk, guard, calls


def create(sdk, **overrides):
    return sdk.responses.create(
        **{"model": MODEL, "input": "OK", "reasoning": {"effort": "none"}, "max_output_tokens": 16,
           **overrides},
    )


def generations(calls):
    return sum(path.endswith("/responses") for path, _ in calls)


def test_probe_only_when_free_balance_unknown():
    sdk, guard, calls = setup()
    with sdk:
        create(sdk)
        assert guard.report["payer"] == "openai"
        assert guard.report["reserved_tokens"] == 26
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert generations(calls) == 1
    assert len(calls) == 2  # No additional count request either.
    assert "balance unknown" in guard.report["stop_reason"]


@pytest.mark.parametrize("billing", [None, {}, {"payer": "developer"}, {"payer": "unexpected"}])
def test_non_openai_or_unknown_payer_blocks_all_followups(billing):
    sdk, guard, calls = setup(remaining=1000, response=raw_response(billing=billing))
    with sdk:
        create(sdk)
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert generations(calls) == 1 and len(calls) == 2
    assert guard.report["stop_reason"]


@pytest.mark.parametrize("usage", [
    None, {}, {"input_tokens": 5, "output_tokens": 3},
    {"input_tokens": 5, "output_tokens": 3, "total_tokens": 9},
    {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
    {"input_tokens": 5, "output_tokens": 17, "total_tokens": 22},
    {"input_tokens": -1, "output_tokens": 3, "total_tokens": 2},
    {"input_tokens": True, "output_tokens": 3, "total_tokens": 4},
])
def test_bad_usage_retains_reservation_and_blocks_followups(usage):
    sdk, guard, calls = setup(remaining=1000, response=raw_response(usage=usage))
    with sdk:
        create(sdk)
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert generations(calls) == 1 and len(calls) == 2
    assert guard.report["reserved_tokens"] == 26
    assert guard.report["total_tokens"] == 0


def test_total_budget_checked_before_generation_even_when_balance_is_large():
    counts = iter([10, TOKEN_BUDGET - 16])
    sdk, guard, calls = setup(remaining=1_000_000, count=lambda: next(counts))
    with sdk:
        create(sdk)
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert generations(calls) == 1
    assert guard.report["reserved_tokens"] == 26
    assert "Total token budget" in guard.report["stop_reason"]


def test_exact_total_budget_boundary_is_allowed_without_refunding_unused_output():
    counts = iter([10, TOKEN_BUDGET - 26 - 16, 10])
    sdk, guard, calls = setup(remaining=1_000_000, count=lambda: next(counts))
    with sdk:
        create(sdk)
        create(sdk)
        assert guard.report["reserved_tokens"] == TOKEN_BUDGET
        assert guard.report["total_tokens"] == 16
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert generations(calls) == 2
    assert guard.report["reserved_tokens"] == TOKEN_BUDGET


def test_concurrent_workers_cannot_overspend_verified_balance():
    sdk, guard, calls = setup(remaining=80)

    def attempt(_):
        try:
            create(sdk)
            return True
        except APIConnectionError:
            return False

    with sdk:
        create(sdk)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
    assert sum(results) == 2
    assert generations(calls) == 3
    assert guard.report["reserved_tokens"] == 78
    assert guard.report["free_tokens_remaining"] == 2
    assert guard.report["total_tokens"] == 24  # No cache/reasoning double counting.


def test_transport_failure_blocks_sdk_retries_and_keeps_reservation():
    sdk, guard, calls = setup(remaining=1000, retries=1, generation_error=True)
    with sdk, pytest.raises(APIConnectionError):
        create(sdk)
    assert generations(calls) == 1 and len(calls) == 2
    assert guard.report["reserved_tokens"] == 26
    assert "transport failed" in guard.report["stop_reason"]


def test_payer_switch_blocks_already_queued_workers_before_network_use():
    responses = iter([raw_response(), raw_response(billing={"payer": "developer"})])
    sdk, guard, calls = setup(remaining=1000, response=lambda: next(responses))

    def attempt(_):
        try:
            create(sdk)
            return True
        except APIConnectionError:
            return False

    with sdk:
        create(sdk)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
    assert sum(results) == 1  # The payer-switching response is returned and accounted.
    assert generations(calls) == 2 and len(calls) == 4
    assert guard.report["payer"] == "developer"
    assert guard.report["reserved_tokens"] == 52


@pytest.mark.parametrize("failure", ["count_http", "count_transport", "count_json", "generation_json"])
def test_unreadable_or_failed_preflight_and_response_stop_future_sends(failure):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            if failure == "count_transport":
                raise httpx.ReadTimeout("simulated")
            if failure == "count_http":
                return httpx.Response(500, json={"input_tokens": 10})
            if failure == "count_json":
                return httpx.Response(200, content=b"not JSON")
            return httpx.Response(200, json={"input_tokens": 10})
        return httpx.Response(200, content=b"not JSON")

    guard = LiveBudgetTransport(httpx.MockTransport(handler), free_tokens_remaining=1000)
    with OpenAI(api_key="test-placeholder", max_retries=0, http_client=httpx.Client(transport=guard)) as sdk:
        try:
            create(sdk)
        except (APIConnectionError, ValueError):
            pass
        initial_calls = len(calls)
        with pytest.raises(APIConnectionError):
            create(sdk)
    assert guard.report["stop_reason"]
    assert len(calls) == initial_calls
    assert guard.report["reserved_tokens"] == (26 if failure == "generation_json" else 0)


@pytest.mark.parametrize("count", [None, -1, True, "10", 129])
def test_invalid_or_large_probe_count_prevents_generation(count):
    sdk, guard, calls = setup(count=count)
    with sdk, pytest.raises(APIConnectionError):
        create(sdk)
    assert generations(calls) == 0
    assert guard.report["reserved_tokens"] == 0


@pytest.mark.parametrize("kwargs", [
    {"model": "gpt-6-sol"}, {"input": "longer probe"}, {"max_output_tokens": 256},
    {"reasoning": {"effort": "high"}}, {"stream": True},
])
def test_unsafe_probe_payload_is_rejected_without_network(kwargs):
    sdk, guard, calls = setup()
    with sdk, pytest.raises(APIConnectionError):
        create(sdk, **kwargs)
    assert calls == []
    assert guard.report["stop_reason"]


def test_sdk_parse_failure_still_accounts_usage_and_counts_schema_before_send():
    class Answer(BaseModel):
        answer: str

    raw = raw_response()
    raw["output"] = [{
        "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": '{"answer":', "annotations": []}],
    }]
    sdk, guard, calls = setup(remaining=1000, response=raw)
    with sdk:
        create(sdk)
        result = RailClient.openai(client=sdk).responses.parse(
            model=MODEL, input="Set answer to OK", reasoning_effort="none", max_output_tokens=16,
            text_format=Answer,
        )
    assert result.error and "ValidationError" in result.error
    assert guard.report["total_tokens"] == 16
    assert guard.report["reserved_tokens"] == 52
    assert calls[2][1]["text"] == calls[3][1]["text"]
    assert calls[2][1]["text"]["format"]["type"] == "json_schema"
