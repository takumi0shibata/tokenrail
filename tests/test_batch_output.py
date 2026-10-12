from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tokenrail
from tokenrail import BatchExecutor, BatchItem, NormalizedResponse, UsageBreakdown
from tokenrail.types import CostBreakdown, TimingBreakdown


class FakeAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return NormalizedResponse(
            id=kwargs["request_id"],
            model="fake-model",
            provider="fake",
            output_text="日本語の結果",
            raw_response={"id": "resp_api", "status": "completed", "extra": {"preserved": True}},
            output_parsed={"answer": "ok"},
            usage=UsageBreakdown(input_tokens=2, output_tokens=3, total_tokens=5),
            billing={"payer": "developer"},
            cost=CostBreakdown(nominal_usd=0.1, developer_usd=0.1, payer="developer"),
            timing=TimingBreakdown(started_at=10, completed_at=11, latency_seconds=1),
            metadata={"purpose": "test"},
            status="completed",
        )


def client(api):
    return SimpleNamespace(responses=api, provider=SimpleNamespace(name="fake"))


def batch():
    return [BatchItem("query-1", {"input": "hello", "model": "fake-model"})]


def test_output_path_is_required():
    with pytest.raises(TypeError, match="output_path"):
        BatchExecutor(client=client(FakeAPI()))


@pytest.mark.parametrize("path", [None, ""])
def test_invalid_output_path_is_rejected(path):
    api = FakeAPI()
    with pytest.raises(ValueError, match="output_path"):
        BatchExecutor(client=client(api), output_path=path)
    assert api.calls == []


@pytest.mark.parametrize("as_string", [False, True])
def test_default_output_preserves_full_response_and_resumes(tmp_path, as_string):
    path = tmp_path / "nested" / "results.jsonl"
    api = FakeAPI()
    executor = BatchExecutor(client=client(api), output_path=str(path) if as_string else path, progress=False)
    assert executor.run(batch()).success_requests == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == FakeAPI().create(request_id="query-1").to_dict()
    assert record["id"] == "query-1"
    assert record["raw_response"]["id"] == "resp_api"
    assert "日本語の結果" in path.read_text(encoding="utf-8")
    calls = len(api.calls)
    assert executor.run(batch()).skipped_requests == 1
    assert BatchExecutor(client=client(api), output_path=path, progress=False).run(batch()).skipped_requests == 1
    assert len(api.calls) == calls
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("projected", [{}, {"id": "wrong", "error": "invented", "status": "failed"}])
def test_projector_selects_fields_with_authoritative_metadata(tmp_path, projected):
    path = tmp_path / "results.jsonl"
    api = FakeAPI()
    executor = BatchExecutor(
        client=client(api),
        output_path=path,
        progress=False,
        projector=lambda response: {"text": response.output_text, **projected},
    )
    executor.run(batch())
    assert json.loads(path.read_text()) == {
        "text": "日本語の結果", "id": "query-1", "error": None, "status": "completed"
    }
    assert executor.run(batch()).skipped_requests == 1
    assert len(api.calls) == 1


@pytest.mark.parametrize("failure", ["projector", "serialization", "not_a_dict", "open", "fsync"])
def test_failed_save_propagates_and_is_retryable(tmp_path, failure):
    path = tmp_path / "results.jsonl"
    previous = '{"id":"prior"}\n'
    path.write_text(previous)
    api = FakeAPI()

    def project(response):
        if failure == "projector":
            raise RuntimeError("projector failed")
        if failure == "serialization":
            return {"unsupported": object()}
        if failure == "not_a_dict":
            return None
        return {"text": response.output_text}

    executor = BatchExecutor(client=client(api), output_path=path, projector=project, progress=False)
    if failure == "open":
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            # Allow resume lookup; fail specifically when appending the result.
            with patch.object(executor._writer, "load_done_ids", return_value={"prior"}):
                with pytest.raises(OSError, match="disk full"):
                    executor.run(batch())
    elif failure == "fsync":
        with patch("tokenrail._storage.os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                executor.run(batch())
    else:
        with pytest.raises(RuntimeError if failure == "projector" else TypeError):
            executor.run(batch())
    assert path.read_text() == previous
    assert len(api.calls) == 1
    recovered = BatchExecutor(client=client(api), output_path=path, progress=False)
    assert recovered.run(batch()).success_requests == 1
    assert recovered.run(batch()).skipped_requests == 1
    assert len(api.calls) == 2


def test_progress_only_controls_display(tmp_path, capsys):
    snapshots = []
    for progress in (True, False):
        executor = BatchExecutor(
            client=client(FakeAPI()), output_path=tmp_path / f"{progress}.jsonl", progress=progress
        )
        with patch("tokenrail._monitor.time.time", return_value=100):
            snapshots.append(executor.run(batch()).to_dict())
        output = capsys.readouterr().out
        if progress:
            assert "tokenrail · 1 requests" in output
            assert "Done 1/1" in output
        else:
            assert output == ""
    assert snapshots[0] == snapshots[1]


def test_empty_batch_still_creates_output_file(tmp_path):
    path = tmp_path / "results.jsonl"
    executor = BatchExecutor(client=client(FakeAPI()), output_path=path, progress=False)
    assert executor.run([]).total_requests == 0
    assert path.read_text() == ""


def test_query_ids_are_not_used_as_file_names(tmp_path):
    path = tmp_path / "results.jsonl"
    executor = BatchExecutor(client=client(FakeAPI()), output_path=path, progress=False)
    executor.run([BatchItem("../NUL", {"input": "hello"})])
    assert json.loads(path.read_text())["id"] == "../NUL"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("name", ["ResultsJsonlSink", "PerRequestJsonSink", "RollingMetricsMonitor"])
def test_previous_classes_are_not_public(name):
    assert name not in tokenrail.__all__
    assert not hasattr(tokenrail, name)


@pytest.mark.parametrize("kwargs", [{"sinks": []}, {"monitor": None}])
def test_previous_executor_arguments_are_rejected(tmp_path, kwargs):
    with pytest.raises(TypeError, match="unexpected keyword"):
        BatchExecutor(client=client(FakeAPI()), output_path=tmp_path / "results.jsonl", **kwargs)
