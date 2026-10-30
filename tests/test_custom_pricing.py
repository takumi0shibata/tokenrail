from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

from tokenrail import BatchExecutor, BatchItem, RailClient
from tokenrail._monitor import _format_usd, _RollingMetricsMonitor
from tokenrail._pricing import _PricingResolver
from tokenrail.catalog import ModelCatalogFallbackWarning, ModelPricing
from tokenrail.types import CostBreakdown, NormalizedResponse

D = Decimal
PRICE = ModelPricing(D('2'), D('0.5'), D('4'), cache_write_input_per_million=D('3'))
QUESTIONS = [{'type': 'predicate', 'name': 'q', 'instructions': 'Evaluate.'}]
USAGE = {'input_tokens': 1500, 'input_tokens_details': {'cached_tokens': 500, 'cache_write_tokens': 700},
         'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': 1510}


class Answer(BaseModel):
    answer: str


def sdk(calls, endpoint='https://resource.openai.azure.com/openai/v1/', usage=None):
    def handler(request):
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        raw = {'model': 'gpt-6-luna', 'usage': USAGE if usage is None else usage}
        if request.url.path.endswith('/decisions'):
            raw['answers'] = [{'name': 'q', 'type': 'predicate', 'probability': 0.8}]
        else:
            raw.update(id='resp_test', object='response', created_at=0, status='completed',
                       parallel_tool_calls=True, tool_choice='auto', tools=[],
                       output=[{'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                                'content': [{'type': 'output_text', 'text': '{"answer":"yes"}', 'annotations': []}]}])
        return httpx.Response(200, json=raw)
    return OpenAI(api_key='mock-only', base_url=endpoint, max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def call(rail, api, model):
    args = {'model': model, 'input': 'test'}
    if api == 'decisions.create':
        return rail.decisions.create(**args, questions=QUESTIONS)
    if api == 'responses.parse':
        return rail.responses.parse(**args, text_format=Answer)
    return rail.responses.create(**args)


@pytest.mark.parametrize('api', ['responses.create', 'responses.parse', 'decisions.create'])
def test_explicit_prices_win_preserve_wire_and_save_provenance(api, tmp_path):
    model = 'prod-gpt-6-luna'
    family = api.split('.')[0]
    config = {model: {family: PRICE}}
    calls = []
    with sdk(calls) as client, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rail = RailClient.openai(client=client, pricing=config)
        config[model][family] = replace(PRICE, input_per_million=D('999'))
        result = call(rail, api, model)
    assert not caught
    assert result.error is None
    assert result.cost.nominal_usd == pytest.approx(.00299)
    assert calls[0][1]['model'] == model
    assert 'pricing' not in calls[0][1]
    assert calls[0][0].endswith('/decisions' if family == 'decisions' else '/responses')
    assert result.model == 'gpt-6-luna'  # Response model does not override the deployment's price.
    evidence = result.to_dict()['cost']['pricing']
    assert evidence['source'] == 'explicit' and evidence['api'] == api
    assert evidence['requested_model'] == model
    assert evidence['rates']['input_per_million'] == '2'
    assert 'azure.com' not in json.dumps(evidence) and 'mock-only' not in json.dumps(evidence)
    if api == 'responses.parse':
        assert result.output_parsed.answer == 'yes'


@pytest.mark.parametrize('model,api,matched', [
    ('prod-gpt-4o-mini-east', 'responses.create', 'gpt-4o-mini'),
    ('prod/gpt-6-luna:blue', 'decisions.create', 'gpt-6-luna'),
    ('gpt-6-luna-2026-10-01', 'decisions.create', 'gpt-6-luna'),
    ('abcgpt-6-lunaxyz', 'decisions.create', None),
    ('gpt-6-sol_gpt-6-luna', 'decisions.create', 'gpt-6-luna'),
    ('gpt-5.4-mini_gpt-5.4-nano', 'responses.create', None),
    ('gpt-4o-mini', 'decisions.create', None),
    ('unknown', 'responses.create', None),
])
def test_longest_delimited_match_ambiguity_and_api_support(model, api, matched):
    calls = []
    with sdk(calls) as client, pytest.warns(ModelCatalogFallbackWarning):
        result = call(RailClient.openai(client=client), api, model)
    assert calls[0][1]['model'] == model
    if matched is None:
        assert result.cost is None
    else:
        assert result.cost.pricing['matched_model'] == matched
        assert result.cost.pricing['source'] == 'openai_catalog'


def test_api_prices_are_separate_and_warnings_once_per_client_model_operation():
    calls = []
    with sdk(calls) as client, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rail = RailClient.openai(client=client)
        with ThreadPoolExecutor(max_workers=4) as pool:
            decisions = list(pool.map(lambda _: call(rail, 'decisions.create', 'prod-gpt-6-luna'), range(8)))
        responses = call(rail, 'responses.create', 'prod-gpt-6-luna')
        call(rail, 'responses.create', 'prod-gpt-6-luna')
        call(rail, 'responses.parse', 'prod-gpt-6-luna')
        call(RailClient.openai(client=client), 'decisions.create', 'prod-gpt-6-luna')
    assert len(caught) == 4
    text = str(caught[0].message)
    for fragment in ['api=decisions.create', "matched model='gpt-6-luna'", 'OpenAI catalog',
                     'does not guarantee', 'pricing=', 'ModelPricing']:
        assert fragment in text
    assert decisions[0].cost.nominal_usd == pytest.approx(.0001225)
    assert responses.cost.nominal_usd == pytest.approx(.0001275)


@pytest.mark.parametrize('field', ['input_per_million', 'cached_input_per_million',
                                   'output_per_million', 'cache_write_input_per_million'])
@pytest.mark.parametrize('value', [D('-1'), D('NaN'), D('Infinity'), True])
def test_invalid_rates_fail_before_any_request(field, value):
    calls = []
    with sdk(calls) as client, pytest.raises(ValueError, match='finite non-negative'):
        RailClient.openai(client=client, pricing={'deployment': {'decisions': replace(PRICE, **{field: value})}})
    assert calls == []


@pytest.mark.parametrize('field', ['cached_input_per_million', 'cache_write_input_per_million'])
def test_missing_cache_rate_is_unknown_but_explicit_zero_is_valid(field):
    with sdk([]) as client:
        rail = RailClient.openai(client=client, pricing={'deployment': {'decisions': replace(PRICE, **{field: None})}})
        with pytest.warns(ModelCatalogFallbackWarning, match='required cache pricing is missing'):
            assert call(rail, 'decisions.create', 'deployment').cost is None
        zero = RailClient.openai(client=client, pricing={'deployment': {'decisions': replace(PRICE, **{field: D(0)})}})
        result = call(zero, 'decisions.create', 'deployment')
        expected = .00274 if field == 'cached_input_per_million' else .00089
        assert result.cost.nominal_usd == pytest.approx(expected)


def test_unused_cache_rates_can_be_omitted_and_zero_price_preserved():
    usage = {'input_tokens': 1, 'output_tokens': 0, 'total_tokens': 1}
    price = ModelPricing(D(0), None, D(0))
    with sdk([], usage=usage) as client:
        rail = RailClient.openai(client=client, pricing={'deployment': {'decisions': price}})
        assert call(rail, 'decisions.create', 'deployment').cost.nominal_usd == 0


@pytest.mark.parametrize('pricing', [[], {'': {}}, {'dep': []}, {'dep': {'response': PRICE}},
                                     {'dep': {'decisions': {}}}])
def test_invalid_config_is_rejected(pricing):
    with pytest.raises((ValueError, TypeError)):
        _PricingResolver(pricing)


def test_standard_openai_keeps_prices_payer_and_sends_no_pricing_options():
    with sdk([], endpoint='https://api.openai.com/v1/') as client, warnings.catch_warnings(record=True) as caught:
        rail = RailClient.openai(client=client)
        assert call(rail, 'decisions.create', 'gpt-6-luna').cost.nominal_usd == pytest.approx(.0001225)
        assert call(rail, 'responses.create', 'gpt-6-luna').cost.nominal_usd == pytest.approx(.0001275)
    assert not caught
    old = CostBreakdown(nominal_usd=1, developer_usd=1)
    assert old.to_dict() == {'nominal_usd': 1, 'developer_usd': 1, 'openai_usd': 0, 'payer': None}


def test_batch_projector_preserves_evidence_and_unknown_subtotal(tmp_path, capsys):
    with sdk([]) as client, pytest.warns(ModelCatalogFallbackWarning):
        path = tmp_path / 'results.jsonl'
        executor = BatchExecutor(client=RailClient.openai(client=client), output_path=path, max_workers=1,
                                 projector=lambda r: {'cost': r.cost.to_dict() if r.cost else None})
        stats = executor.run([BatchItem(m, {'model': m, 'input': 'test', 'questions': QUESTIONS})
                              for m in ['prod-gpt-6-luna', 'unknown']])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]['cost']['pricing']['matched_model'] == 'gpt-6-luna'
    assert records[1]['cost'] is None
    assert stats.unpriced_requests == 1 and stats.nominal_usd > 0
    assert 'Known subtotal <$0.01' in capsys.readouterr().out


@pytest.mark.parametrize('amount,expected', [(0, '$0.00'), (.00000001, '<$0.01'), (.00999, '<$0.01'),
                                            (.01, '$0.01'), (12.345, '$12.35')])
def test_display_money_boundaries(amount, expected):
    assert _format_usd(amount) == expected


def test_all_unknown_is_never_displayed_as_free():
    monitor = _RollingMetricsMonitor(printer=None)
    monitor.start(total_requests=1, todo_requests=1, skipped_requests=0)
    snap = monitor.record(NormalizedResponse(id='1', model='deployment', provider='openai',
                                           output_text=None, raw_response=None))
    assert '$— (oai — / dev $—)' in monitor.format_summary(snap)
    assert 'Total $— · 1 unpriced' in monitor.format_final(snap)


def test_both_api_overrides_and_omitted_api_fallback():
    with sdk([]) as client, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rail = RailClient.openai(client=client, pricing={'gpt-6-luna': {
            'responses': PRICE, 'decisions': replace(PRICE, output_per_million=D(0)),
        }})
        assert call(rail, 'responses.create', 'gpt-6-luna').cost.nominal_usd == pytest.approx(.00299)
        assert call(rail, 'decisions.create', 'gpt-6-luna').cost.nominal_usd == pytest.approx(.00295)
        assert not caught
        partial = RailClient.openai(client=client, pricing={'gpt-6-luna': {'responses': PRICE}})
        assert call(partial, 'decisions.create', 'gpt-6-luna').cost.nominal_usd == pytest.approx(.0001225)
        assert len(caught) == 1


def test_prices_do_not_override_capability_validation():
    calls = []
    with sdk(calls) as client:
        rail = RailClient.openai(client=client, pricing={'gpt-6-astra': {'responses': PRICE}})
        with pytest.raises(ValueError, match='temperature is not supported'):
            rail.responses.create(model='gpt-6-astra', input='test', temperature=1)
    assert calls == []
