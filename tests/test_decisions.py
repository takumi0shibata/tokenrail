from __future__ import annotations

import copy
import json

import httpx
import openai
import pytest
from openai import OpenAI
from pydantic import BaseModel

from tokenrail import BatchExecutor, BatchItem, RailClient, batch_items_from_queries
from tokenrail.catalog import calculate_cost
from tokenrail.types import UsageBreakdown

QUESTIONS = [{"type": "predicate", "name": "refund", "instructions": "Does the customer request a refund?"}]


class Summary(BaseModel):
    answer: str


def decision_body(answers=None, **extra):
    return {
        "model": "gpt-6-luna",
        "answers": answers if answers is not None else [{"type": "predicate", "name": "refund", "probability": 0.9}],
        "usage": {
            "input_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 700},
            "output_tokens": 0,
            "total_tokens": 1500,
        },
        **extra,
    }


def sdk_client(handler, base_url="https://api.openai.com/v1/"):
    return OpenAI(
        api_key="mock-only",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_decisions_preserve_all_answer_types_usage_raw_body_and_metadata():
    answers = [
        {"type": "predicate", "name": "refund", "probability": 0.9},
        {"type": "choice", "name": "team", "choice": "billing", "confidence": 0.8,
         "probabilities": [{"value": "billing", "probability": 0.9}, {"value": "other", "probability": 0.1}]},
        {"type": "score", "name": "severity", "score": 0.7, "confidence": 0.6,
         "probabilities": [{"value": 0, "label": "low", "probability": 0.3},
                           {"value": 1, "label": "high", "probability": 0.7}]},
        {"type": "refusal", "name": "declined"},
    ]
    questions = [
        *QUESTIONS,
        {"type": "choice", "name": "team", "instructions": "Choose a team.",
         "choices": [{"value": "billing"}, {"value": "other"}]},
        {"type": "score", "name": "severity", "instructions": "Rate severity.",
         "levels": [{"label": "low"}, {"label": "high"}]},
        {"type": "predicate", "name": "declined", "instructions": "Declined question."},
    ]
    calls = []

    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=decision_body(answers, billing={"payer": "openai"}, custom="kept"))

    with sdk_client(handler) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(
            model="gpt-6-luna", input="Refund please.", questions=questions,
            request_id="ticket", metadata={"source": "test"},
        )
    assert calls == [("/v1/decisions", {"model": "gpt-6-luna", "input": "Refund please.", "questions": questions})]
    assert response.id == "ticket"
    assert response.api == "decisions.create"
    assert response.error is None and response.status == "completed"
    assert response.output_text is None and response.output_parsed is None
    assert response.answers == answers
    assert response.to_dict()["answers"] == answers
    assert response.raw_response["custom"] == "kept"
    assert response.metadata == {"source": "test"}
    assert response.usage.input_tokens == 1500
    assert response.usage.cached_tokens == 500 and response.usage.cache_write_tokens == 700
    assert response.usage.total_tokens == 1500
    assert response.cost.nominal_usd == pytest.approx(0.0001225)
    assert response.cost.openai_usd == response.cost.nominal_usd
    assert response.cost.developer_usd == 0
    assert response.timing.latency_seconds >= 0


def test_decisions_forward_json_images_cache_settings_and_sdk_options_without_mutation():
    image = {"type": "input_image", "image_url": "https://example.test/photo.png"}
    inputs = [{"role": "user", "content": [
        {"type": "input_text", "text": "Evidence", "prompt_cache_breakpoint": {"mode": "explicit"}},
        {"type": "input_json", "data": {"charges": [10, 10]}}, image,
    ]}]
    questions = [{**QUESTIONS[0], "instructions": [{"type": "input_text", "text": "Compare."}, image]}]
    kwargs = dict(
        model="gpt-6-luna", input=inputs, questions=questions,
        prompt_cache_key="reference", prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        safety_identifier="customer", extra_body={"custom": "pass-through"},
        extra_headers={"X-Check": "forwarded"}, timeout=10,
    )
    original = copy.deepcopy(kwargs)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert request.headers["X-Check"] == "forwarded"
        return httpx.Response(200, json=decision_body())

    with sdk_client(handler) as sdk:
        result = RailClient.openai(client=sdk).decisions.create(**kwargs)
    assert result.error is None
    expected = {key: value for key, value in kwargs.items() if key not in {"timeout", "extra_headers", "extra_body"}}
    assert calls == [expected | {"custom": "pass-through"}]
    assert kwargs == original


@pytest.mark.parametrize("kwargs", [
    {"questions": None}, {"questions": []}, {"questions": {}}, {"questions": "invalid"},
    {"questions": [None]}, {"questions": QUESTIONS * 401},
    {"questions": QUESTIONS, "text_format": Summary},
    {"questions": QUESTIONS, "extra_body": {"text_format": None}},
    {"questions": QUESTIONS, "extra_body": {"questions": []}},
    {"questions": QUESTIONS, "prompt_cache_key": "a", "extra_body": {"prompt_cache_key": "b"}},
    {"questions": QUESTIONS, "prompt_cache_options": {"mode": "explicit"},
     "extra_body": {"prompt_cache_options": {"mode": "implicit"}}},
    {"questions": QUESTIONS, "extra_body": {"model": "different"}},
    {"questions": QUESTIONS, "extra_body": []},
    {"questions": QUESTIONS, "stream": True},
    {"questions": QUESTIONS, "extra_body": {"background": True}},
])
def test_invalid_decision_requests_do_not_send(kwargs):
    calls = []
    with sdk_client(lambda request: calls.append(request)) as sdk:
        with pytest.raises(ValueError):
            RailClient.openai(client=sdk).decisions.create(model="gpt-6-luna", input="test", **kwargs)
    assert calls == []


@pytest.mark.parametrize("kwargs", [
    {"questions": []}, {"questions": None}, {"extra_body": {"questions": []}},
    {"questions": QUESTIONS, "text_format": Summary},
    {"extra_body": {"questions": QUESTIONS}, "text_format": Summary},
])
def test_invalid_routing_is_detected_before_any_batch_request(tmp_path, kwargs):
    calls = []
    with sdk_client(lambda request: calls.append(request)) as sdk:
        executor = BatchExecutor(client=RailClient.openai(client=sdk), output_path=tmp_path / "out.jsonl")
        with pytest.raises(ValueError):
            executor.run([
                BatchItem("valid", {"model": "gpt-6-luna", "input": "test"}),
                BatchItem("invalid", {"model": "gpt-6-luna", "input": "test", **kwargs}),
            ])
    assert calls == []


@pytest.mark.parametrize("method", ["create", "parse"])
@pytest.mark.parametrize("in_body", [False, True])
def test_explicit_responses_namespace_rejects_questions(method, in_body):
    calls = []
    kwargs = {"extra_body": {"questions": QUESTIONS}} if in_body else {"questions": QUESTIONS}
    if method == "parse":
        kwargs["text_format"] = Summary
    with sdk_client(lambda request: calls.append(request)) as sdk:
        with pytest.raises(ValueError, match="decisions.create"):
            getattr(RailClient.openai(client=sdk).responses, method)(model="gpt-6-luna", input="test", **kwargs)
    assert calls == []


@pytest.mark.parametrize("model", ["gpt-6-luna", "my-deployment"])
@pytest.mark.parametrize("endpoint", [
    "https://resource.openai.azure.com/openai/v1/", "https://proxy.test/v1/",
])
def test_custom_endpoints_preserve_deployment_names_and_do_not_guess_prices(endpoint, model):
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)["model"]))
        return httpx.Response(200, json=decision_body(model=model))

    with sdk_client(handler, base_url=endpoint) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(model=model, input="test", questions=QUESTIONS)
    assert calls == [(endpoint + "decisions", model)]
    assert response.error is None and response.cost is None
    assert response.usage.total_tokens == 1500


def test_costs_use_endpoint_rates_and_do_not_charge_decision_output():
    usage = UsageBreakdown(input_tokens=1500, cached_tokens=500, cache_write_tokens=700, output_tokens=1000)
    decisions = calculate_cost("gpt-6-luna", usage, "developer", api="decisions.create")
    responses = calculate_cost("gpt-6-luna", usage, "developer")
    assert decisions.nominal_usd == pytest.approx(0.0001225)
    assert responses.nominal_usd == pytest.approx(0.0006225)
    assert calculate_cost("gpt-4o", usage, "developer", api="decisions.create") is None
    assert calculate_cost("my-gpt-6-luna-deployment", usage, "developer", api="decisions.create") is None


def test_mixed_batch_routes_logs_saves_and_resumes_each_operation(tmp_path, capsys):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/decisions"):
            return httpx.Response(200, json=decision_body())
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0, "model": body["model"],
            "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "output": [{"id": "msg", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"answer":"ok"}', "annotations": []}]}],
        })

    items = [
        BatchItem("decision", {"model": "gpt-6-luna", "input": "test", "questions": QUESTIONS}),
        BatchItem("body-decision", {"model": "gpt-6-luna", "input": "test", "extra_body": {"questions": QUESTIONS}}),
        BatchItem("parsed", {"model": "gpt-6-luna", "input": "test", "text_format": Summary}),
        BatchItem("text", {"model": "gpt-6-luna", "input": "test"}),
    ]
    original = copy.deepcopy(items)
    path = tmp_path / "results.jsonl"
    with sdk_client(handler) as sdk:
        executor = BatchExecutor(client=RailClient.openai(client=sdk), output_path=path, max_workers=3)
        stats = executor.run(items)
        assert stats.success_requests == 4 and stats.error_requests == 0
        assert stats.input_tokens == 3002 and stats.total_tokens == 3006
        assert stats.nominal_usd == pytest.approx(0.0002472)
        records = {record["id"]: record for record in map(json.loads, path.read_text().splitlines())}
        assert executor.run(items).skipped_requests == 4
    assert items == original
    assert len(calls) == 4
    assert sum(path == "/v1/decisions" for path, _ in calls) == 2
    assert records["decision"]["api"] == records["body-decision"]["api"] == "decisions.create"
    assert records["parsed"]["api"] == "responses.parse"
    assert records["text"]["api"] == "responses.create"
    assert records["decision"]["answers"][0]["probability"] == 0.9
    assert records["parsed"]["output_parsed"] == {"answer": "ok"}
    log = capsys.readouterr().out
    for operation in ("decisions.create", "responses.parse", "responses.create"):
        assert f"api={operation}" in log


@pytest.mark.parametrize("failure", ["http_error", "missing_answers", "answer_count", "api_error"])
def test_failed_decisions_preserve_api_and_retry_on_resume(tmp_path, capsys, failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "http_error":
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        raw = decision_body()
        if failure == "missing_answers":
            raw.pop("answers")
        elif failure == "answer_count":
            raw["answers"] = []
        else:
            raw["error"] = {"message": "failed"}
        return httpx.Response(200, json=raw)

    path = tmp_path / "out.jsonl"
    items = batch_items_from_queries({"job": "test"}, model="gpt-6-luna", questions=QUESTIONS)
    with sdk_client(handler) as sdk:
        executor = BatchExecutor(client=RailClient.openai(client=sdk), output_path=path)
        assert executor.run(items).error_requests == 1
        assert executor.run(items).error_requests == 1
    assert len(calls) == 2
    for record in map(json.loads, path.read_text().splitlines()):
        assert record["api"] == "decisions.create" and record["error"]
        if failure != "http_error":
            assert record["raw_response"] is not None and record["usage"]["input_tokens"] == 1500
    log = capsys.readouterr().out
    assert "ERR" in log and "api=decisions.create" in log


def test_question_refusals_are_saved_and_not_automatically_reissued(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=decision_body([{"type": "refusal", "name": "refund"}]))

    path = tmp_path / "out.jsonl"
    # The helper materializes a shared one-shot questions iterable before building items.
    items = batch_items_from_queries({"a": "test", "b": "test"}, model="gpt-6-luna", questions=iter(QUESTIONS))
    with sdk_client(handler) as sdk:
        executor = BatchExecutor(client=RailClient.openai(client=sdk), output_path=path, progress=False)
        assert executor.run(items).success_requests == 2
        assert executor.run(items).skipped_requests == 2
    assert len(calls) == 2
    for record in map(json.loads, path.read_text().splitlines()):
        assert record["answers"] == [{"type": "refusal", "name": "refund"}]
        assert record["error"] is None


@pytest.mark.parametrize("field", ["choices", "levels", "instructions"])
@pytest.mark.parametrize("container", [list, tuple, iter], ids=["list", "tuple", "iterator"])
@pytest.mark.parametrize("in_body", [False, True])
def test_shared_question_iterables_survive_fanout_and_resume(tmp_path, field, container, in_body):
    values = {
        "choices": [{"value": "a"}, {"value": "b"}],
        "levels": [{"label": "low"}, {"label": "high"}],
        "instructions": [{"type": "input_text", "text": "Evaluate the input."}],
    }[field]
    original = container(values)
    question = {
        "type": {"choices": "choice", "levels": "score", "instructions": "predicate"}[field],
        "name": "q", "instructions": "Evaluate the input.", field: original,
    }
    questions = iter([question])
    shared = {"extra_body": {"questions": questions}} if in_body else {"questions": questions}
    items = batch_items_from_queries({"a": "first", "b": "second"}, model="gpt-6-luna", **shared)
    assert question[field] is original
    assert (shared["extra_body"] if in_body else shared)["questions"] is questions
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(503, json={"error": {"message": "temporary", "type": "server_error"}})
        return httpx.Response(200, json=decision_body([{"name": "q", "type": "refusal"}]))

    # Block instructions became a public SDK input in 3.37. Earlier versions
    # cannot serialize an instructions iterator; the helper must not add support.
    old_instruction_iterator = (
        field == "instructions" and container is iter
        and tuple(map(int, openai.__version__.split(".")[:2])) < (3, 37)
    )
    with sdk_client(handler) as sdk:
        executor = BatchExecutor(
            client=RailClient.openai(client=sdk), output_path=tmp_path / "out.jsonl",
            max_workers=1, progress=False,
        )
        first = executor.run(items)
        if old_instruction_iterator:
            assert first.error_requests == 2 and calls == []
            records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
            assert all("not JSON serializable" in record["error"] for record in records)
            return
        assert first.success_requests == 1 and first.error_requests == 1
        resumed = executor.run(items)
        assert resumed.success_requests == 1 and resumed.skipped_requests == 1
        assert executor.run(items).skipped_requests == 2
    expected = {**question, field: values}
    assert [body["questions"] for body in calls] == [[expected]] * 3
    assert [body["input"] for body in calls] == ["first", "second", "second"]


def test_shared_question_preparation_leaves_unrelated_json_and_strings_untouched():
    opaque_iterator = iter([1, 2])
    custom = {"values": opaque_iterator}
    question = {"type": "predicate", "name": "q", "instructions": "Evaluate.", "custom": custom}
    before = dict(question)
    items = batch_items_from_queries({"a": "input"}, model="gpt-6-luna", questions=[question])
    prepared = items[0].request_kwargs["questions"][0]
    assert prepared == before and question == before
    assert prepared["custom"] is custom
    assert next(opaque_iterator) == 1


@pytest.mark.parametrize("answer", [
    None, {}, {"type": "unknown", "name": "refund"},
    {"type": "predicate", "name": "refund"},
    {"type": "predicate", "name": "refund", "probability": None},
    {"type": "predicate", "name": "refund", "probability": 1.1},
    {"type": "predicate", "name": "other", "probability": 0.9},
    {"type": "refusal", "name": "other"},
    {"type": "score", "name": "refund", "score": 0.7, "confidence": 0.9, "probabilities": []},
])
def test_malformed_decision_answers_preserve_evidence_and_remain_retryable(tmp_path, answer):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=decision_body([answer]))

    path = tmp_path / "out.jsonl"
    items = batch_items_from_queries({"job": "test"}, model="gpt-6-luna", questions=QUESTIONS)
    with sdk_client(handler) as sdk:
        executor = BatchExecutor(client=RailClient.openai(client=sdk), output_path=path, progress=False)
        for _ in range(2):
            stats = executor.run(items)
            assert stats.error_requests == 1 and stats.success_requests == 0
            assert stats.skipped_requests == 0
            assert stats.total_tokens == 1500
    assert len(calls) == 2
    for record in map(json.loads, path.read_text().splitlines()):
        assert record["error"] and "answer" in record["error"].lower()
        assert record["raw_response"]["answers"] == record["answers"]
        assert record["usage"]["total_tokens"] == 1500
        assert record["cost"]["nominal_usd"] > 0


@pytest.mark.parametrize("question,answer", [
    ({"type": "choice", "choices": [{"value": False}, {"value": "other"}]},
     {"type": "choice", "choice": False, "confidence": 0.9,
      "probabilities": [{"value": False, "probability": 0.9}, {"value": "other", "probability": 0.1}]}),
    ({"type": "score", "levels": [{"label": "low"}, {"label": "high"}]},
     {"type": "score", "score": 0.7, "confidence": 0.8,
      "probabilities": [{"value": 0, "label": "low", "probability": 0.3},
                        {"value": 1, "label": "high", "probability": 0.7}]}),
])
def test_decision_answer_required_fields_and_valid_controls(question, answer):
    question = {**question, "instructions": "Classify this input."}
    # Every required field must survive validation, including boolean choice values.
    cases = [answer, {"type": "refusal"}, *[
        {key: value for key, value in answer.items() if key != missing}
        for missing in answer
    ]]
    with sdk_client(lambda request: httpx.Response(200, json=decision_body([cases.pop(0)]))) as sdk:
        client = RailClient.openai(client=sdk)
        for index in range(2 + len(answer)):
            response = client.decisions.create(model="gpt-6-luna", input="test", questions=[question])
            if index < 2:
                assert response.error is None and response.status == "completed"
            else:
                assert response.error and response.usage.total_tokens == 1500


@pytest.mark.parametrize("probability", [0, 1, 0.9, True, "0.9"])
def test_decision_validation_preserves_sdk_numeric_normalization(probability):
    # The SDK coerces numeric strings and booleans before tokenrail sees the answer.
    answer = {"type": "predicate", "name": "refund", "probability": probability}
    with sdk_client(lambda request: httpx.Response(200, json=decision_body([answer]))) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(
            model="gpt-6-luna", input="test", questions=QUESTIONS,
        )
    assert response.error is None
    assert response.answers[0]["probability"] == float(probability)


@pytest.mark.parametrize("field,value", [
    ("confidence", None), ("confidence", -0.1), ("probabilities", None), ("probabilities", []),
    ("probabilities", [None]), ("probabilities", [{"value": "yes"}]),
    ("probabilities", [{"value": "yes", "probability": 1.1}]),
])
def test_decision_distributions_reject_missing_or_invalid_values(field, value):
    answer = {"type": "choice", "choice": "yes", "confidence": 0.9,
              "probabilities": [{"value": "yes", "probability": 1.0}], field: value}
    with sdk_client(lambda request: httpx.Response(200, json=decision_body([answer]))) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(
            model="gpt-6-luna", input="test", questions=[{
                "type": "choice", "instructions": "Choose.", "choices": [{"value": "yes"}],
            }],
        )
    assert response.error and response.usage.total_tokens == 1500


def test_score_validation_allows_fractional_scores_and_one_shot_levels():
    answer = {"type": "score", "score": 1.1, "confidence": 0.55, "probabilities": [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.7},
        {"value": 2, "label": "high", "probability": 0.2},
    ]}
    with sdk_client(lambda request: httpx.Response(200, json=decision_body([answer]))) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(
            model="gpt-6-luna", input="test", questions=[{
                "type": "score", "instructions": "Rate severity.",
                "levels": ({"label": label} for label in ("low", "medium", "high")),
            }],
        )
    assert response.error is None and response.answers[0]["score"] == 1.1


def test_projector_cannot_remove_or_replace_decision_api(tmp_path):
    path = tmp_path / "out.jsonl"
    with sdk_client(lambda request: httpx.Response(200, json=decision_body())) as sdk:
        executor = BatchExecutor(
            client=RailClient.openai(client=sdk), output_path=path, progress=False,
            projector=lambda result: {"answers": result.answers, "api": "responses.parse"},
        )
        executor.run(batch_items_from_queries({"job": "test"}, model="gpt-6-luna", questions=QUESTIONS))
    record = json.loads(path.read_text())
    assert record["api"] == "decisions.create"
    assert record["answers"] == decision_body()["answers"]


def test_azure_batch_reports_unpriced_results_instead_of_a_zero_cost_total(tmp_path, capsys):
    items = batch_items_from_queries({"job": "test"}, model="gpt-6-luna", questions=QUESTIONS)
    with sdk_client(
        lambda request: httpx.Response(200, json=decision_body()),
        base_url="https://resource.openai.azure.com/openai/v1/",
    ) as sdk:
        stats = BatchExecutor(
            client=RailClient.openai(client=sdk), output_path=tmp_path / "out.jsonl",
        ).run(items)
    assert stats.unpriced_requests == stats.to_dict()["unpriced_requests"] == 1
    assert stats.by_model["gpt-6-luna"].unpriced_requests == 1
    assert stats.input_tokens == 1500
    assert "Total $— · 1 unpriced" in capsys.readouterr().out


def test_mixed_priced_and_unpriced_results_show_known_subtotal(tmp_path, capsys):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=decision_body(model=body["model"]))

    with sdk_client(handler) as sdk:
        stats = BatchExecutor(
            client=RailClient.openai(client=sdk), output_path=tmp_path / "out.jsonl",
        ).run([
            BatchItem("known", {"model": "gpt-6-luna", "input": "test", "questions": QUESTIONS}),
            BatchItem("unknown", {"model": "deployment", "input": "test", "questions": QUESTIONS}),
        ])
    assert len(calls) == 2
    assert stats.unpriced_requests == 1
    assert stats.nominal_usd == pytest.approx(0.0001225)
    log = capsys.readouterr().out
    assert "Known subtotal" in log and "1 unpriced" in log
    unknown_model_line = next(line for line in log.splitlines() if line.startswith("  deployment"))
    assert "$—" in unknown_model_line
