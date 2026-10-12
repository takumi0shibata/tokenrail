"""Opt-in, bounded live API checks. Never enabled by the normal test command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from _live_budget import MODEL, LiveBudgetTransport
from openai import OpenAI
from pydantic import BaseModel, field_validator

from tokenrail import BatchExecutor, BatchItem, RailClient

pytestmark = pytest.mark.skipif(
    os.getenv("TOKENRAIL_RUN_LIVE_TESTS") != "1", reason="Set TOKENRAIL_RUN_LIVE_TESTS=1 to spend API tokens"
)


class Answer(BaseModel):
    answer: str


class RejectingAnswer(Answer):
    @field_validator("answer")
    @classmethod
    def reject_answer(cls, value: str) -> str:
        raise ValueError("intentional local validation failure")


@pytest.fixture(scope="module")
def live():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required for live tests")
    remaining = os.getenv("TOKENRAIL_LIVE_FREE_TOKENS_REMAINING")
    guard = LiveBudgetTransport(
        httpx.HTTPTransport(retries=0), free_tokens_remaining=int(remaining) if remaining else None,
    )
    report = guard.report
    try:
        with OpenAI(
            base_url="https://api.openai.com/v1", timeout=60, max_retries=0,
            http_client=httpx.Client(transport=guard),
        ) as sdk:
            client = RailClient.openai(client=sdk)
            # The only generation allowed without a verified free-token balance.
            client.responses.create(input="OK", **kwargs(max_output_tokens=16))
            yield client, report
    finally:
        destination = os.getenv("TOKENRAIL_LIVE_REPORT")
        if destination:
            Path(destination).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Live usage:", json.dumps(report))


@pytest.fixture(scope="module")
def live_ready(live):
    _, report = live
    if report["stop_reason"] or report["free_tokens_remaining"] is None:
        pytest.skip(report["stop_reason"] or "Verified free-token balance required after the initial probe")
    return live


def test_live_payer_probe(live):
    _, report = live
    assert report["requests"] == 1
    assert report["payer"] == "openai"
    assert report["stop_reason"] is None
    assert report["total_tokens"] > 0


def kwargs(**extra):
    return {
        "model": MODEL,
        "reasoning_effort": "none",
        "max_output_tokens": 256,
        "prompt_cache_options": {"mode": "explicit"},
        **extra,
    }


def test_live_create_and_parse(live_ready):
    client, _ = live_ready
    response = client.responses.create(input="Reply with exactly TOKENRAIL_OK.", temperature=0, **kwargs())
    assert response.status == "completed" and response.error is None
    assert response.output_text == "TOKENRAIL_OK"
    parsed = client.responses.parse(input="Set answer to TOKENRAIL_OK.", text_format=Answer, **kwargs())
    assert parsed.error is None
    assert parsed.output_parsed.answer == "TOKENRAIL_OK"
    assert parsed.usage.total_tokens > 0
    assert parsed.cost is not None


def test_live_parallel_batch_resume_and_id_preflight(live_ready, tmp_path):
    client, report = live_ready
    path = tmp_path / "results.jsonl"
    batch = [BatchItem(str(i), kwargs(input="Set answer to TOKENRAIL_OK.", text_format=Answer)) for i in range(3)]
    executor = BatchExecutor(client=client, output_path=path, max_workers=2, progress=False)
    stats = executor.run(batch)
    assert stats.success_requests == 3 and stats.error_requests == 0
    assert stats.total_tokens > 0
    requests = report["requests"]
    assert executor.run(batch).skipped_requests == 3
    with pytest.raises(ValueError, match="must not be empty"):
        executor.run([*batch, BatchItem("", kwargs(input="hello", request_id=""))])
    assert report["requests"] == requests
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {record["id"] for record in records} == {"0", "1", "2"}
    assert all(record["raw_response"]["status"] == "completed" for record in records)


def test_live_incomplete_structured_output(live_ready):
    client, report = live_ready
    requests = report["requests"]
    result = client.responses.parse(
        input="Set answer to a detailed essay of at least 400 words about prime numbers.",
        text_format=Answer,
        **kwargs(max_output_tokens=32),
    )
    assert report["requests"] == requests + 1
    assert result.status == "incomplete" and result.error
    assert result.output_parsed is None
    assert result.output_text
    assert result.raw_response and result.usage.total_tokens > 0
    assert result.cost is not None


def test_live_local_validation_failure_is_retryable(live_ready, tmp_path):
    client, report = live_ready
    path = tmp_path / "results.jsonl"
    executor = BatchExecutor(client=client, output_path=path, max_workers=1, progress=False)
    batch = [BatchItem("invalid", kwargs(input="Set answer to TOKENRAIL_OK.", text_format=RejectingAnswer))]
    for _ in range(2):
        requests = report["requests"]
        stats = executor.run(batch)
        assert report["requests"] == requests + 1
        assert stats.error_requests == 1 and stats.success_requests == 0
        assert stats.total_tokens > 0
        assert stats.nominal_usd > 0
        assert stats.skipped_requests == 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    record = records[-1]
    assert record["status"] == "completed"
    assert "intentional local validation failure" in record["error"]
    assert record["raw_response"]["status"] == "completed"
    assert record["output_text"]
    assert record["usage"]["total_tokens"] > 0


def test_live_explicit_prompt_cache(live_ready):
    client, _ = live_ready
    reference = "Reference entry: prime numbers have exactly two positive divisors. " * 300
    shared = {"type": "input_text", "text": reference, "prompt_cache_breakpoint": {"mode": "explicit"}}
    key = f"tokenrail-live-{uuid4().hex}"
    for _ in range(2):
        result = client.responses.create(
            input=[
                {"role": "developer", "content": [shared]},
                {"role": "user", "content": "Reply with exactly TOKENRAIL_OK."},
            ],
            prompt_cache_key=key,
            **kwargs(max_output_tokens=64),
        )
        assert result.error is None and result.status == "completed"
        assert result.usage.input_tokens >= 1024
        assert result.usage.cached_tokens >= 0 and result.usage.cache_write_tokens >= 0
        assert result.cost is not None
