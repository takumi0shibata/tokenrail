from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

from tokenrail import BatchExecutor, BatchItem, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.catalog import ModelCatalogFallbackWarning, calculate_cost
from tokenrail.types import NormalizedResponse, UsageBreakdown


def response(item_id="job"):
    return NormalizedResponse(id=item_id, model="gpt-4o", provider="fake", output_text="ok", raw_response=None)


class FakeAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return response(kwargs["request_id"])


def executor(api, sinks):
    return BatchExecutor(
        client=SimpleNamespace(responses=api, provider=SimpleNamespace(name="fake")),
        sinks=sinks,
        max_workers=1,
        monitor=RollingMetricsMonitor(printer=None),
    )


def items(**kwargs):
    return [BatchItem("job", {"model": "gpt-4o", "input": "hello", **kwargs})]


@pytest.mark.parametrize(
    "item_id", ["../victim", "/tmp/victim", "..\\victim", "", ".", "..", "a\x00b", "C:evil", "NUL"]
)
def test_unsafe_ids_cannot_write_or_send_requests(tmp_path, item_id):
    sink = PerRequestJsonSink(tmp_path / "out")
    victim = tmp_path / "victim.json"
    victim.write_text("original")
    with pytest.raises(ValueError, match="Unsafe"):
        sink.save(response(item_id))
    api = FakeAPI()
    with pytest.raises(ValueError, match="must not be empty" if not item_id else "Unsafe"):
        executor(api, [sink]).run([BatchItem(item_id, {"model": "gpt-4o", "input": "hello"})])
    assert victim.read_text() == "original"
    assert api.calls == []
    assert list(sink.output_dir.iterdir()) == []


def test_destination_symlink_is_replaced_without_touching_target(tmp_path):
    sink = PerRequestJsonSink(tmp_path / "out")
    victim = tmp_path / "victim.json"
    victim.write_text('{"id":"outside"}')
    destination = sink.output_dir / "job.json"
    destination.symlink_to(victim)
    assert sink.load_done_ids() == set()
    sink.save(response())
    assert victim.read_text() == '{"id":"outside"}'
    assert not destination.is_symlink()
    assert sink.load_done_ids() == {"job"}


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_atomic_save_failure_preserves_previous_record_and_cleans_tempfiles(tmp_path, failure):
    sink = PerRequestJsonSink(tmp_path)
    sink.save(response())
    previous = (tmp_path / "job.json").read_bytes()
    changed = response()
    changed.output_text = "changed"
    with patch(f"tokenrail.sinks.os.{failure}", side_effect=OSError("disk failure")):
        with pytest.raises(OSError):
            sink.save(changed)
    assert (tmp_path / "job.json").read_bytes() == previous
    assert list(tmp_path.iterdir()) == [tmp_path / "job.json"]


def test_corrupt_and_unfinished_per_request_files_are_retried(tmp_path):
    sink = PerRequestJsonSink(tmp_path)
    for name, data in {"job": b"{", "empty": b"{}", "scalar": b"[]", "unicode": b"\xff"}.items():
        (tmp_path / f"{name}.json").write_bytes(data)
    (tmp_path / "failed.json").write_text('{"id":"resp","status":"failed"}')
    assert sink.load_done_ids() == set()
    api = FakeAPI()
    stats = executor(api, [sink]).run(items())
    assert stats.success_requests == 1
    assert len(api.calls) == 1
    assert sink.load_done_ids() == {"job"}


@pytest.mark.parametrize("tail", [b'{"id":"broken"', b'{"text":"\xe3', b"garbage", b"[]"])
def test_jsonl_repairs_broken_tail_and_keeps_prior_records(tmp_path, tail):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"id":"prior"}\n' + tail)
    sink = ResultsJsonlSink(path)
    sink.save(response())
    assert sink.load_done_ids() == {"prior", "job"}
    assert [json.loads(line)["id"] for line in path.read_bytes().splitlines()] == ["prior", "job"]


@pytest.mark.parametrize("prefix", [b"", b'{"id":"prior"}\n'])
def test_jsonl_preserves_complete_tail_without_newline(tmp_path, prefix):
    path = tmp_path / "results.jsonl"
    path.write_bytes(prefix + b'{"id":"tail"}')
    sink = ResultsJsonlSink(path)
    sink.save(response())
    expected = {"tail", "job"} | ({"prior"} if prefix else set())
    assert sink.load_done_ids() == expected


def test_jsonl_ignores_non_object_and_invalid_utf8_records(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'[]\nnull\n1\n\xff\n{"id":"good"}\n')
    assert ResultsJsonlSink(path).load_done_ids() == {"good"}


@pytest.mark.parametrize("projected", [{}, {"error": None, "status": "completed"}])
def test_projector_cannot_hide_failure_from_resume(tmp_path, projected):
    sink = ResultsJsonlSink(tmp_path / "results.jsonl", projector=lambda r: {"id": r.id, **projected})
    failed = response()
    failed.status = "incomplete"
    failed.error = "max_output_tokens"
    sink.save(failed)
    assert sink.load_done_ids() == set()
    record = json.loads(sink.path.read_text())
    assert record["status"] == "incomplete"
    assert record["error"] == "max_output_tokens"


def test_first_atomic_save_failure_leaves_no_completed_file(tmp_path):
    sink = PerRequestJsonSink(tmp_path)
    with patch("tokenrail.sinks.os.fsync", side_effect=OSError("disk failure")):
        with pytest.raises(OSError):
            sink.save(response())
    assert sink.load_done_ids() == set()
    assert list(tmp_path.iterdir()) == []


class Shape(BaseModel):
    answer: str


def raw_response(status):
    return {
        "id": "resp_test",
        "model": "gpt-4o",
        "created_at": 0,
        "object": "response",
        "status": status,
        "output": [],
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
        "error": {"code": "server_error", "message": "failed"} if status == "failed" else None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
    }


@pytest.mark.parametrize("status", ["failed", "incomplete", "queued", "in_progress", "cancelled"])
@pytest.mark.parametrize("method", ["create", "parse"])
def test_sdk_unfinished_responses_are_errors_and_retryable(tmp_path, status, method):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=raw_response(status)))
    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=transport)) as sdk:
        client = RailClient.openai(client=sdk)
        per_request = PerRequestJsonSink(tmp_path / "out")
        jsonl = ResultsJsonlSink(tmp_path / "results.jsonl")
        batch = items(**({"text_format": Shape} if method == "parse" else {}))
        ex = BatchExecutor(client=client, sinks=[jsonl, per_request], monitor=RollingMetricsMonitor(printer=None))
        stats = ex.run(batch)
        assert stats.success_requests == 0
        assert stats.error_requests == 1
        assert stats.input_tokens == 5
        assert jsonl.load_done_ids() == per_request.load_done_ids() == set()
        saved = json.loads(jsonl.path.read_text())
        assert saved["status"] == status
        assert saved["error"]
        assert ex.run(batch).processed_requests == 1


@pytest.mark.parametrize("method", ["create", "parse"])
@pytest.mark.parametrize("extra_body", [False, True])
def test_background_requests_are_rejected_before_network_use(method, extra_body):
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request))
    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=transport)) as sdk:
        client = RailClient.openai(client=sdk)
        kwargs = {"extra_body": {"background": True}} if extra_body else {"background": True}
        if method == "parse":
            kwargs["text_format"] = Shape
        with pytest.raises(ValueError, match="background=True"):
            getattr(client.responses, method)(model="gpt-4o", input="hello", **kwargs)
        assert calls == []


def test_completed_sdk_response_still_succeeds(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=raw_response("completed")))
    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=transport)) as sdk:
        sink = ResultsJsonlSink(tmp_path / "results.jsonl")
        ex = BatchExecutor(
            client=RailClient.openai(client=sdk), sinks=[sink], monitor=RollingMetricsMonitor(printer=None)
        )
        assert ex.run(items()).success_requests == 1
        assert ex.run(items()).skipped_requests == 1


def test_create_preserves_api_refusal():
    raw = raw_response("completed")
    raw["output"] = [
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "Cannot comply"}],
        }
    ]
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=raw))
    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=transport)) as sdk:
        result = RailClient.openai(client=sdk).responses.create(model="gpt-4o", input="hello")
        assert result.refusal == "Cannot comply"
        assert result.status == "completed"


def test_default_timeout_preserves_sdk_default_and_explicit_timeout_is_forwarded():
    with OpenAI(api_key="audit-placeholder") as sdk:
        default_timeout = sdk.timeout
    client = RailClient.openai(api_key="audit-placeholder")
    explicit = RailClient.openai(api_key="audit-placeholder", timeout=12.0)
    try:
        assert client.provider._client.timeout == default_timeout
        assert default_timeout is not None
        assert explicit.provider._client.timeout == 12.0
    finally:
        client.provider._client.close()
        explicit.provider._client.close()


def test_gpt4o_may_snapshot_uses_its_own_official_price_without_fallback_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost = calculate_cost(
            "gpt-4o-2024-05-13", UsageBreakdown(input_tokens=1_000_000, output_tokens=1_000_000), payer="developer"
        )
    assert cost.nominal_usd == 20.0
    assert not any(w.category is ModelCatalogFallbackWarning for w in caught)
    assert calculate_cost("gpt-4o", UsageBreakdown(input_tokens=1_000_000), payer="developer").nominal_usd == 2.5


def test_partial_multi_sink_save_is_recovered_on_rerun(tmp_path):
    api = FakeAPI()
    first = ResultsJsonlSink(tmp_path / "results.jsonl")
    second = PerRequestJsonSink(tmp_path / "out")
    ex = executor(api, [first, second])
    with patch.object(second, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            ex.run(items())
    assert first.load_done_ids() == {"job"}
    assert second.load_done_ids() == set()
    assert ex.run(items()).success_requests == 1
    assert first.load_done_ids() == second.load_done_ids() == {"job"}
    assert ex.run(items()).skipped_requests == 1
    assert len(api.calls) == 2


@pytest.mark.parametrize("request_id", ["different", None, ""])
def test_mismatched_request_ids_fail_before_any_batch_request(tmp_path, request_id):
    api = FakeAPI()
    sink = ResultsJsonlSink(tmp_path / "results.jsonl")
    batch = [BatchItem("valid", {"model": "gpt-4o", "input": "hello"}), *items(request_id=request_id)]
    with pytest.raises(ValueError, match="request_id must match"):
        executor(api, [sink]).run(batch)
    assert api.calls == []
    assert sink.path.read_text() == ""


def test_matching_request_id_is_resumable(tmp_path):
    api = FakeAPI()
    ex = executor(api, [ResultsJsonlSink(tmp_path / "results.jsonl")])
    assert ex.run(items(request_id="job")).success_requests == 1
    assert ex.run(items(request_id="job")).skipped_requests == 1
    assert len(api.calls) == 1


@pytest.mark.parametrize("sink_kind", ["none", "jsonl", "per_request"])
@pytest.mark.parametrize("batch_kind", ["sequence", "mapping"])
def test_empty_batch_id_is_rejected_before_any_request(tmp_path, sink_kind, batch_kind):
    api = FakeAPI()
    sinks = []
    if sink_kind == "jsonl":
        sinks = [ResultsJsonlSink(tmp_path / "results.jsonl")]
    elif sink_kind == "per_request":
        sinks = [PerRequestJsonSink(tmp_path / "out")]
    if batch_kind == "sequence":
        batch = [
            BatchItem("valid", {"model": "gpt-4o", "input": "hello"}),
            BatchItem("", {"model": "gpt-4o", "input": "hello", "request_id": ""}),
        ]
    else:
        batch = {"valid": "hello", "": "hello"}
    with pytest.raises(ValueError, match="BatchItem.id must not be empty"):
        executor(api, sinks).run(batch)
    assert api.calls == []
    if sinks:
        assert sinks[0].load_done_ids() == set()


def structured_raw(status, text):
    raw = raw_response(status)
    raw["output"] = [
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]
    raw["usage"] = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    raw["billing"] = {"payer": "developer"}
    return raw


@pytest.mark.parametrize(
    "status,text",
    [
        ("incomplete", '{"answer":'),
        ("completed", '{"answer":'),
        ("completed", '{"wrong_field":"ok"}'),
    ],
)
def test_direct_parse_failure_preserves_response_without_another_request(status, text):
    raw = structured_raw(status, text)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        result = RailClient.openai(client=sdk).responses.parse(
            model="gpt-4o",
            input="hello",
            text_format=Shape,
            request_id="job",
            metadata={"purpose": "test"},
        )
    assert len(calls) == 1
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert result.id == "job"
    assert result.status == status
    assert result.error and "ValidationError" in result.error
    if status == "incomplete":
        assert "max_output_tokens" in result.error
    assert result.output_text == text
    assert result.output_parsed is None
    assert result.raw_response == raw
    assert result.usage.total_tokens == 110
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.cost.nominal_usd == pytest.approx(0.00035)
    assert result.cost.developer_usd == pytest.approx(0.00035)
    assert result.billing == {"payer": "developer"}
    assert result.metadata == {"purpose": "test"}
    assert result.timing.completed_at >= result.timing.started_at


@pytest.mark.parametrize("status", ["incomplete", "completed"])
@pytest.mark.parametrize("sink_kind", ["jsonl", "per_request", "both"])
def test_batch_parse_failure_keeps_usage_and_retries_until_success(tmp_path, status, sink_kind):
    raw = structured_raw(status, '{"answer":')
    successful = structured_raw("completed", '{"answer":"ok"}')
    payloads = [raw, successful]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payloads.pop(0))

    jsonl = ResultsJsonlSink(tmp_path / "results.jsonl")
    per_request = PerRequestJsonSink(tmp_path / "out")
    sinks = {"jsonl": [jsonl], "per_request": [per_request], "both": [jsonl, per_request]}[sink_kind]
    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        batch = items(text_format=Shape)
        ex = BatchExecutor(
            client=RailClient.openai(client=sdk), sinks=sinks, monitor=RollingMetricsMonitor(printer=None)
        )
        failed = ex.run(batch)
        assert len(calls) == 1
        assert failed.success_requests == 0
        assert failed.error_requests == 1
        assert failed.total_tokens == 110
        assert failed.nominal_usd == pytest.approx(0.00035)
        for sink in sinks:
            assert sink.load_done_ids() == set()
            path = sink.path if isinstance(sink, ResultsJsonlSink) else sink.output_dir / "job.json"
            record = json.loads(path.read_text())
            assert record["status"] == status
            assert "ValidationError" in record["error"]
            assert record["output_text"] == '{"answer":'
            assert record["usage"]["total_tokens"] == 110
            assert record["cost"]["nominal_usd"] == pytest.approx(0.00035)
            if isinstance(sink, PerRequestJsonSink):
                assert record["raw_response"] == raw
        assert ex.run(batch).success_requests == 1
        assert len(calls) == 2
        for sink in sinks:
            assert sink.load_done_ids() == {"job"}
        assert ex.run(batch).skipped_requests == 1
        assert len(calls) == 2
    if jsonl in sinks:
        records = [json.loads(line) for line in jsonl.path.read_text().splitlines()]
        assert records[1]["output_parsed"] == {"answer": "ok"}
        assert records[1]["error"] is None


@pytest.mark.parametrize("mode", ["parsed", "refusal", "incomplete_valid_json"])
def test_raw_response_path_preserves_success_refusal_and_incomplete_status(mode):
    status = "incomplete" if mode == "incomplete_valid_json" else "completed"
    raw = structured_raw(status, '{"answer":"ok"}')
    if mode == "refusal":
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot comply"}]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=raw)

    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        result = RailClient.openai(client=sdk).responses.parse(model="gpt-4o", input="hello", text_format=Shape)
    assert len(calls) == 1
    assert result.raw_response == raw
    assert result.status == status
    assert result.usage.total_tokens == 110
    assert result.cost.nominal_usd == pytest.approx(0.00035)
    if mode == "refusal":
        assert result.refusal == "Cannot comply"
        assert result.output_parsed is None
        assert result.error is None
    else:
        assert isinstance(result.output_parsed, Shape)
        assert result.output_parsed.answer == "ok"
        if mode == "incomplete_valid_json":
            assert result.error and "max_output_tokens" in result.error
        else:
            assert result.error is None


def test_raw_response_path_does_not_swallow_http_errors_or_resubmit():
    from openai import BadRequestError

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid request", "type": "invalid_request_error"}})

    with OpenAI(api_key="audit-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        with pytest.raises(BadRequestError):
            RailClient.openai(client=sdk).responses.parse(model="gpt-4o", input="hello", text_format=Shape)
    assert len(calls) == 1
