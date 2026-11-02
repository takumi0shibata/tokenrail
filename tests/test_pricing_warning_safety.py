from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest
from test_custom_pricing import PRICE, QUESTIONS, USAGE, Answer, call, sdk

from tokenrail import BatchExecutor, BatchItem, RailClient
from tokenrail.catalog import ModelCatalogFallbackWarning


@pytest.mark.parametrize('api', ['responses.create', 'responses.parse', 'decisions.create'])
@pytest.mark.parametrize('model', ['prod-gpt-6-luna', 'unknown'])
def test_strict_configuration_warning_prevents_every_send(api, model):
    calls = []
    with sdk(calls) as client, warnings.catch_warnings():
        warnings.simplefilter('error', ModelCatalogFallbackWarning)
        rail = RailClient.openai(client=client)
        for _ in range(2):
            with pytest.raises(ModelCatalogFallbackWarning):
                call(rail, api, model)
        assert calls == []


@pytest.mark.parametrize('api', ['responses.create', 'responses.parse', 'decisions.create'])
@pytest.mark.parametrize('field', ['cached_input_per_million', 'cache_write_input_per_million'])
def test_post_response_warning_keeps_usage_and_does_not_resend(api, field, tmp_path, caplog):
    calls = []
    path = tmp_path / 'results.jsonl'
    price = replace(PRICE, **{field: None})
    kwargs = {'model': 'deployment', 'input': 'test'}
    if api == 'decisions.create':
        kwargs['questions'] = QUESTIONS
    elif api == 'responses.parse':
        kwargs['text_format'] = Answer
    with sdk(calls) as client, warnings.catch_warnings():
        warnings.simplefilter('error', ModelCatalogFallbackWarning)
        for attempt in range(2):
            # A fresh client must also skip the completed saved response.
            rail = RailClient.openai(client=client, pricing={'deployment': {api.split('.')[0]: price}})
            batch = BatchExecutor(client=rail, output_path=path, progress=False)
            stats = batch.run([BatchItem('job', kwargs)])
            if attempt == 0:
                assert stats.success_requests == 1 and stats.error_requests == 0
                assert stats.total_tokens == USAGE['total_tokens']
                assert stats.unpriced_requests == 1
            else:
                assert stats.skipped_requests == 1
    assert len(calls) == 1
    row = json.loads(path.read_text())
    assert row['error'] is None and row['status'] == 'completed'
    assert row['raw_response']['usage']['total_tokens'] == USAGE['total_tokens']
    assert row['usage']['cached_tokens'] == USAGE['input_tokens_details']['cached_tokens']
    assert row['cost'] is None
    assert 'required cache pricing is missing' in caplog.text
    if api == 'decisions.create':
        assert row['answers'][0]['probability'] == .8
    else:
        assert row['output_text'] == '{"answer":"yes"}'


def test_strict_post_response_warning_is_reported_once_per_client(caplog):
    with sdk([]) as client, warnings.catch_warnings():
        warnings.simplefilter('error', ModelCatalogFallbackWarning)
        rail = RailClient.openai(client=client, pricing={
            'deployment': {'decisions': replace(PRICE, cached_input_per_million=None)},
        })
        for _ in range(2):
            result = call(rail, 'decisions.create', 'deployment')
            assert result.error is None and result.usage.total_tokens == USAGE['total_tokens']
            assert result.cost is None
    assert len(caplog.records) == 1
