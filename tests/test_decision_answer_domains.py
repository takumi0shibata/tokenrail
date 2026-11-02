from __future__ import annotations

import copy
import json

import httpx
import pytest
from test_decisions import decision_body, sdk_client

from tokenrail import BatchExecutor, BatchItem, RailClient

CHOICE = {'type': 'choice', 'name': 'q', 'instructions': 'Choose.', 'choices': [{'value': 'A'}, {'value': 'B'}]}
SCORE = {'type': 'score', 'name': 'q', 'instructions': 'Rate.', 'levels': [{'label': 'low'}, {'label': 'high'}]}
CHOICE_ANSWER = {'type': 'choice', 'name': 'q', 'choice': 'A', 'confidence': 1.,
                 'probabilities': [{'value': 'A', 'probability': 1.}]}
SCORE_ANSWER = {'type': 'score', 'name': 'q', 'score': .7, 'confidence': .8,
                'probabilities': [{'value': 0, 'label': 'low', 'probability': .3},
                                  {'value': 1, 'label': 'high', 'probability': .7}]}


@pytest.mark.parametrize('question,answer', [
    (CHOICE, {**CHOICE_ANSWER, 'choice': 'C'}),
    (CHOICE, {**CHOICE_ANSWER, 'probabilities': [{'value': 'C', 'probability': 1.}]}),
    (CHOICE, {**CHOICE_ANSWER, 'choice': False}),
    (SCORE, {**SCORE_ANSWER, 'score': -0.1}),
    (SCORE, {**SCORE_ANSWER, 'score': 1.1}),
    (SCORE, {**SCORE_ANSWER, 'probabilities': [{'value': 2, 'label': 'high', 'probability': 1.}]}),
    (SCORE, {**SCORE_ANSWER, 'probabilities': [{'value': 0, 'label': 'high', 'probability': 1.}]}),
])
def test_out_of_domain_answers_keep_evidence_and_are_not_resumed(question, answer, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=decision_body([answer]))
    path = tmp_path / 'results.jsonl'
    with sdk_client(handler) as sdk:
        batch = BatchExecutor(client=RailClient.openai(client=sdk), output_path=path, progress=False)
        items = [BatchItem('job', {'model': 'gpt-6-luna', 'input': 'test', 'questions': [question]})]
        for _ in range(2):
            stats = batch.run(items)
            assert stats.error_requests == 1 and stats.success_requests == 0
            assert stats.total_tokens == 1500 and stats.nominal_usd > 0
            assert stats.skipped_requests == 0
    assert len(calls) == 2
    for row in map(json.loads, path.read_text().splitlines()):
        assert row['error'] and 'does not match' in row['error']
        assert row['answers'] == row['raw_response']['answers'] == [answer]
        assert row['usage']['total_tokens'] == 1500


@pytest.mark.parametrize('question,answer', [
    (CHOICE, CHOICE_ANSWER),
    ({**CHOICE, 'choices': [{'value': False}, {'value': 'false'}]},
     {**CHOICE_ANSWER, 'choice': False, 'probabilities': [{'value': False, 'probability': 1.}]}),
    (SCORE, SCORE_ANSWER),
    (SCORE, {**SCORE_ANSWER, 'score': 0.}),
    (SCORE, {**SCORE_ANSWER, 'score': 1.}),
    (SCORE, {'type': 'refusal', 'name': 'q'}),
])
def test_valid_answers_and_iterable_domains_preserve_request(question, answer):
    expected = copy.deepcopy(question)
    field = 'choices' if question['type'] == 'choice' else 'levels'
    question = {**question, field: iter(copy.deepcopy(question[field]))}
    original_iterator = question[field]
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=decision_body([answer]))
    with sdk_client(handler) as sdk:
        response = RailClient.openai(client=sdk).decisions.create(
            model='gpt-6-luna', input='test', questions=iter([question]),
        )
    assert response.error is None and response.status == 'completed'
    assert response.answers == [answer]
    assert calls[0]['questions'] == [expected]
    assert question[field] is original_iterator  # No mutation of caller dictionaries.
