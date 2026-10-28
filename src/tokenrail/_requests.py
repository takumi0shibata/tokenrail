from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import get_args, get_origin, get_type_hints

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


def _accepts_iterable(annotation: object) -> bool:
    # Include Required/Union wrappers used by the SDK's public TypedDicts.
    return get_origin(annotation) is Iterable or any(_accepts_iterable(arg) for arg in get_args(annotation))


def prepare_shared_decision_request(kwargs: JsonDict) -> JsonDict:
    """Consume shared question iterables once, before creating multiple jobs."""
    from openai.types.decision_create_params import QuestionQuestionParamPredicate

    payload = prepare_decision_request(kwargs)
    # Older SDKs accept only string instructions. Do not extend their contract
    # by converting otherwise unsupported instruction iterators into JSON lists.
    instruction_parts = _accepts_iterable(get_type_hints(QuestionQuestionParamPredicate)["instructions"])
    questions = []
    for original in payload["questions"]:
        question = dict(original)
        kind = question.get("type")
        fields = ["choices"] if kind == "choice" else ["levels"] if kind == "score" else []
        if instruction_parts:
            fields.append("instructions")
        for field in fields:
            value = question.get(field)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                question[field] = list(value)
        questions.append(question)
    payload["questions"] = questions
    return payload
