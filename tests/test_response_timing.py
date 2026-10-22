import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

import tokenrail.providers.openai as provider_module
from tokenrail import BatchExecutor, BatchItem, RailClient


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("route", ["create", "parse", "parse_error", "batch"])
@pytest.mark.parametrize("adjustment", [0, -60, 60])
def test_response_latency_uses_elapsed_time(monkeypatch, tmp_path, route, adjustment):
    clock = {"elapsed": 0.0, "offset": 0.0}
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(
            time=lambda: 10000 + clock["elapsed"] + clock["offset"],
            monotonic=lambda: clock["elapsed"],
        ),
    )
    calls = []

    def handler(request):
        calls.append(request.url.path)
        clock["elapsed"] = 2.0
        clock["offset"] = adjustment
        text = '{"answer":' if route == "parse_error" else '{"answer":"OK"}'
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-6-luna",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                "billing": {"payer": "openai"},
            },
        )

    with OpenAI(
        api_key="mock-only", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as sdk:
        client = RailClient.openai(client=sdk)
        kwargs = dict(model="gpt-6-luna", input="OK")
        if route == "batch":
            path = tmp_path / "batch.jsonl"
            batch = BatchExecutor(client=client, output_path=path, progress=False, max_workers=1)
            assert batch.run([BatchItem("job", kwargs)]).success_requests == 1
            result = json.loads(path.read_text())
        else:
            if route in ("parse", "parse_error"):
                response = client.responses.parse(**kwargs, text_format=Answer)
            else:
                response = client.responses.create(**kwargs)
            result = response.to_dict()
    assert len(calls) == 1
    assert result["usage"]["total_tokens"] == 10
    if route == "parse_error":
        assert result["error"] and "ValidationError" in result["error"]
    else:
        assert result["error"] is None
    assert result["timing"]["started_at"] == 10000
    assert result["timing"]["completed_at"] == 10002 + adjustment
    assert result["timing"]["latency_seconds"] == 2.0
