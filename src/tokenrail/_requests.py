from __future__ import annotations

from collections.abc import Iterable, Mapping

from .types import JsonDict


def request_api(kwargs: JsonDict) -> str:
    """Select the SDK operation from the request, including body extensions."""
    extra_body = kwargs.get("extra_body")
    if "questions" in kwargs or (isinstance(extra_body, dict) and "questions" in extra_body):
        return "decisions.create"
    if kwargs.get("text_format") is not None:
        return "responses.parse"
    return "responses.create"


def prepare_decision_request(kwargs: JsonDict) -> JsonDict:
    """Materialize questions once and reject ambiguous routing before sending."""
    payload = dict(kwargs)
    extra_body = payload.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a dict")
    body = dict(extra_body or {})
    if "text_format" in payload or "text_format" in body:
        raise ValueError("questions and text_format cannot be used together")
    if "questions" in payload and "questions" in body and payload["questions"] != body["questions"]:
        raise ValueError("questions conflicts with extra_body.questions")
    questions = payload.get("questions", body.get("questions"))
    if not isinstance(questions, Iterable) or isinstance(questions, (str, bytes, Mapping)):
        raise ValueError("questions must be a non-empty iterable of question dictionaries")
    questions = list(questions)
    if not 1 <= len(questions) <= 400 or any(not isinstance(question, dict) for question in questions):
        raise ValueError("questions must contain between 1 and 400 question dictionaries")
    payload["questions"] = questions
    # The SDK requires a named questions argument even when it came from extra_body.
    body.pop("questions", None)
    if extra_body is not None:
        payload["extra_body"] = body
    return payload
