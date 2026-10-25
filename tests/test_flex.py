"""Offline Flex discovery, exact routing, fallback, accounting and UI API regressions."""
import copy
import json
import unittest
from unittest.mock import patch, AsyncMock

import httpx

from ctxagent import accounting, flex, llm, providers
from ctxagent.config import CompactionConfig, SessionConfig
from tests.test_productivity import make_session

DIR = [{'name': 'OpenAI', 'slug': 'openai', 'headquarters': 'US', 'datacenters': ['US']},
       {'name': 'Blocked', 'slug': 'blocked', 'headquarters': 'SG', 'datacenters': ['SG']}]
STANDARD = {'provider_name': 'OpenAI', 'tag': 'openai', 'status': 0,
            'pricing': {'prompt': '0.00001', 'completion': '0.00005',
                        'input_cache_read': '0.000001', 'input_cache_write': '0.0000125'}}
FLEX = {'provider_name': 'OpenAI', 'tag': 'openai/flex', 'status': 0,
        'supported_parameters': ['tools', 'reasoning'],
        'pricing': {'prompt': '0.000005', 'completion': '0.000025',
                    'input_cache_read': '0.0000005', 'input_cache_write': '0.00000625'}}
EPS = [STANDARD, FLEX, {**FLEX, 'provider_name': 'Blocked', 'tag': 'blocked/flex'}]


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        flex._CACHE.clear()
        self.enterContext(patch.object(providers, 'providers', AsyncMock(return_value=DIR)))
        self.enterContext(patch.object(providers, 'model_endpoints', AsyncMock(return_value=EPS)))

    async def test_exact_flex_tags_and_region_filter(self):
        result = await flex.discover('openai/gpt-6-astra', 'eu_us')
        self.assertEqual([r['tag'] for r in result['candidates']], ['openai/flex'])
        self.assertEqual(result['candidates'][0]['discount_fractions']['prompt'], .5)
        self.assertEqual(result['candidates'][0]['discount_fractions']['completion'], .5)
        self.assertEqual(result['excluded'][0]['tag'], 'blocked/flex')
        plans = await flex.plans('openai/gpt-6-astra', 'eu_us', 'prefer_flex', needs_tools=True)
        self.assertEqual(plans[0]['provider']['only'], ['openai/flex'])
        self.assertNotIn('order', plans[0]['provider'])
        self.assertTrue(plans[0]['provider']['allow_fallbacks'])
        self.assertEqual(plans[1]['provider']['only'], ['openai'])

    async def test_standard_mode_does_not_probe_or_opt_in_flex(self):
        plan = (await flex.plans('model', 'eu_us', 'standard'))[0]
        providers.model_endpoints.assert_not_awaited()
        payload = llm._apply_route({'model': 'model'}, plan)
        self.assertNotIn('service_tier', payload)
        self.assertEqual(payload['provider']['only'], ['openai'])

    async def test_standard_preview_does_not_use_flex_tariff(self):
        result = await providers.check_model('model', 'eu_us')
        self.assertEqual(len(result['allowed']), 1)
        self.assertEqual(result['allowed'][0]['tag'], 'openai')
        self.assertAlmostEqual(result['allowed'][0]['price_in'], 10)

    async def test_missing_flex_only_refuses_prefer_falls_back(self):
        providers.model_endpoints.return_value = [STANDARD]
        with self.assertRaisesRegex(ValueError, 'No generation sent'):
            await flex.plans('model', 'eu_us', 'flex_only')
        fallback = await flex.plans('model', 'eu_us', 'prefer_flex')
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]['tier'], 'default')
        self.assertIn('standard', fallback[0]['notice'])

    async def test_discovery_failure_no_unrestricted_flex(self):
        providers.model_endpoints.return_value = []
        result = await flex.discover('model', 'eu_us')
        self.assertFalse(result['known'])
        self.assertFalse(result['available'])
        providers.providers.return_value = []
        with self.assertRaisesRegex(ValueError, 'refusing unrestricted'):
            await flex.plans('model', 'eu_us', 'standard')
        self.assertEqual((await flex.plans('model', 'any', 'standard'))[0]['provider'], None)

    async def test_any_policy_still_uses_only_flex_tags(self):
        plans = await flex.plans('model', 'any', 'flex_only')
        self.assertEqual(set(plans[0]['provider']['only']), {'openai/flex','blocked/flex'})
        self.assertEqual(len(plans), 1)

    async def test_cache_refresh_and_tools_filter(self):
        await flex.discover('model', 'eu_us')
        await flex.discover('model', 'eu_us')
        self.assertEqual(providers.model_endpoints.await_count, 1)
        await flex.discover('model', 'eu_us', refresh=True)
        self.assertEqual(providers.model_endpoints.await_count, 2)
        row = flex.catalogue('model', [{**FLEX, 'supported_parameters': []}], DIR, 'eu_us', needs_tools=True)
        self.assertFalse(row['available'])
        self.assertIn('tool use', row['excluded'][0]['reason'])

    def test_dynamic_discounts_not_hardcoded_and_long_context(self):
        e = copy.deepcopy(FLEX)
        e['pricing']['prompt'] = '.000008'
        e['pricing']['overrides'] = [{'min_prompt_tokens': 20000, 'prompt': '.000006'}]
        low = flex.catalogue('model', [STANDARD,e], DIR, 'eu_us', 19999)
        high = flex.catalogue('model', [STANDARD,e], DIR, 'eu_us', 20000)
        self.assertAlmostEqual(low['candidates'][0]['discount_fractions']['prompt'], .2)
        self.assertAlmostEqual(high['candidates'][0]['discount_fractions']['prompt'], .4)
        unknown = flex.catalogue('model', [FLEX], DIR, 'eu_us')
        self.assertIsNone(unknown['candidates'][0]['discount_fractions']['prompt'])

    def test_unknown_provider_and_spoofed_or_fast_tags_excluded(self):
        rows = [FLEX, {**FLEX, 'tag':'openai/fast'}, {**FLEX, 'tag':'other/flex'},
                {**FLEX, 'tag':'openai/flex', 'provider_name':'Unknown'}]
        result = flex.catalogue('model', rows, DIR, 'eu_us')
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(len(result['excluded']), 2)

    def test_modes_roundtrip_and_invalid_rejected(self):
        cfg = SessionConfig.from_dict({'pricing_mode':'prefer_flex',
                                      'compaction': {'summary_pricing_mode':'flex_only'}})
        clone = SessionConfig.from_dict(cfg.to_dict())
        self.assertEqual(clone.pricing_mode, 'prefer_flex')
        self.assertEqual(clone.compaction.summary_pricing_mode, 'flex_only')
        self.assertEqual(SessionConfig.from_dict({}).pricing_mode,'standard')
        with self.assertRaises(ValueError): SessionConfig.from_dict({'pricing_mode':'cheap'})
        with self.assertRaises(ValueError): CompactionConfig.from_dict({'summary_pricing_mode':'cheap'})


def sse_response(tier='flex', error_after=False):
    chunks = [{'id':'gen-test', 'model':'openai/gpt-6-astra', 'provider':'OpenAI', 'service_tier':tier,
               'choices':[{'delta':{'content':'OK'}}]}]
    if error_after:
        chunks.append({'error':{'code':503,'message':'capacity failure'}})
    else:
        chunks.append({'usage':{'prompt_tokens':100, 'completion_tokens':2, 'cost':.00055,
                     'prompt_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
                     'completion_tokens_details':{'reasoning_tokens':0}},
                       'choices':[{'finish_reason':'stop','delta':{}}]})
    return httpx.Response(200, text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        flex._CACHE.clear()
        self.enterContext(patch.object(providers, 'providers', AsyncMock(return_value=DIR)))
        self.enterContext(patch.object(providers, 'model_endpoints', AsyncMock(return_value=EPS)))
        self.enterContext(patch.object(llm, '_headers', return_value={}))

    def mock_http(self, handler):
        client = httpx.AsyncClient
        def factory(**kw): return client(transport=httpx.MockTransport(handler), **kw)
        self.enterContext(patch.object(llm.httpx, 'AsyncClient', factory))

    async def stream(self, mode='prefer_flex'):
        return [e async for e in llm.stream_chat(model='openai/gpt-6-astra', messages=[{'role':'user','content':'hello'}],
                    pricing_mode=mode, session_id='stable-session', frozen_msg_count=1)]

    async def test_success_sends_exact_tag_and_service_tier_and_preserves_cache(self):
        sent=[]
        def handler(request):
            sent.append(json.loads(request.content)); return sse_response()
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),1)
        self.assertEqual(sent[0]['service_tier'],'flex')
        self.assertEqual(sent[0]['provider']['only'],['openai/flex'])
        self.assertEqual(sent[0]['session_id'],'stable-session')
        self.assertIn('prompt_cache_breakpoint',sent[0]['messages'][0]['content'][0])
        usage=next(e for e in events if e['type']=='usage')
        self.assertTrue(usage['routing']['flex_confirmed'])
        self.assertEqual(usage['routing']['reported_service_tier'],'flex')

    async def test_exact_502_sse_capacity_falls_back_before_generation(self):
        sent=[]
        error={'code':502,'message':'Flex processing is temporarily unavailable. Please try again later or use standard processing.',
               'metadata':{'error_type':'provider_unavailable'}}
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200,text='data: '+json.dumps({'error':error})+'\n\n') if len(sent)==1 else sse_response('default')
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),2)
        self.assertEqual(sent[1]['service_tier'],'default')
        self.assertEqual(sent[0]['session_id'],sent[1]['session_id'])
        self.assertEqual(sent[0]['messages'],sent[1]['messages'])
        self.assertEqual(sent[1]['provider']['only'],['openai'])
        self.assertFalse(any(e['type']=='error' for e in events))
        self.assertTrue(any('falling back' in e.get('routing_notice','') for e in events))

    async def test_exact_502_json_body_fallback_and_strict_refusal(self):
        sent=[]
        error={'code':502,'message':'Flex processing is temporarily unavailable.', 'metadata':{'error_type':'provider_unavailable'}}
        def handler(request):
            sent.append(json.loads(request.content))
            if len(sent)==1: return httpx.Response(200,json={'error':error})
            return httpx.Response(200,json={'choices':[{'message':{'content':'standard answer'}}]})
        self.mock_http(handler)
        self.assertEqual(await llm.complete(model='model',messages=[],pricing_mode='prefer_flex'),'standard answer')
        self.assertEqual(len(sent),2)
        sent.clear()
        with self.assertRaises(RuntimeError):
            await llm.complete(model='model',messages=[],pricing_mode='flex_only')
        self.assertEqual(len(sent),1)

    async def test_http200_json_error_stream_falls_back(self):
        sent=[]
        def handler(request):
            sent.append(1)
            if len(sent)==1:
                return httpx.Response(200,json={'error':{'code':502,'message':'Flex capacity unavailable'}})
            return sse_response('default')
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),2)
        self.assertEqual(events[-1]['type'],'done')

    async def test_reasoning_and_tool_partial_error_never_replayed(self):
        for delta in ({'reasoning':'hidden work'},{'reasoning_details':[{'type':'reasoning.text','text':'work'}]},
                      {'tool_calls':[{'index':0,'function':{'name':'write_file','arguments':'{'}}]}):
            sent=[]
            def handler(request):
                sent.append(1)
                chunks=[{'choices':[{'delta':delta}]},{'error':{'code':502,'message':'Flex capacity unavailable'}}]
                return httpx.Response(200,text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks))
            with patch.object(llm.httpx,'AsyncClient',self._partial_factory(handler)):
                events=await self.stream()
            self.assertEqual(len(sent),1)
            self.assertEqual(events[-1]['type'],'error')

    def _partial_factory(self,handler):
        # Capture the real client before any transport patch.
        return lambda **kw: httpx._client.AsyncClient(transport=httpx.MockTransport(handler),**kw)

    async def test_capacity_fallback_is_explicit_same_session_and_region(self):
        sent=[]
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(429,text='Flex capacity exhausted') if len(sent)==1 else sse_response('default')
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),2)
        self.assertEqual(sent[1]['service_tier'],'default')
        self.assertEqual(sent[1]['provider']['only'],['openai'])
        self.assertEqual(sent[0]['messages'],sent[1]['messages'])
        self.assertEqual(sent[0]['session_id'],sent[1]['session_id'])
        self.assertTrue(any('falling back' in e.get('routing_notice','') for e in events))
        self.assertFalse(next(e for e in events if e['type']=='usage')['routing']['flex_confirmed'])

    async def test_flex_only_capacity_error_not_retried(self):
        sent=[]
        def handler(request): sent.append(1); return httpx.Response(429,text='capacity')
        self.mock_http(handler)
        events=await self.stream('flex_only')
        self.assertEqual(len(sent),1)
        self.assertEqual(events[-1]['type'],'error')

    async def test_no_fallback_auth_credit_or_generic_bad_request(self):
        for code in (400,401,402,403):
            self.assertFalse(flex.fallback_allowed(code, 'invalid request'))
        self.assertTrue(flex.fallback_allowed(400,'service_tier flex not supported'))
        sent=[]
        def handler(request): sent.append(1); return httpx.Response(402,text='out of credits')
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),1)
        self.assertEqual(events[-1]['type'],'error')

    async def test_partial_stream_error_never_replayed(self):
        sent=[]
        def handler(request): sent.append(1); return sse_response(error_after=True)
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),1)
        self.assertTrue(any(e['type']=='text' for e in events))
        self.assertEqual(events[-1]['type'],'error')

    async def test_read_timeout_never_replayed(self):
        sent=[]
        def handler(request): sent.append(1); raise httpx.ReadTimeout('uncertain request',request=request)
        self.mock_http(handler)
        with self.assertRaises(httpx.ReadTimeout): await self.stream()
        self.assertEqual(len(sent),1)

    async def test_reported_default_or_missing_does_not_confirm_flex(self):
        self.mock_http(lambda request:sse_response(None))
        events=await self.stream()
        self.assertFalse(next(e for e in events if e['type']=='usage')['routing']['flex_confirmed'])
        self.assertFalse(flex.metadata({'mode':'flex_only','tier':'flex'},'default')['flex_confirmed'])

    async def test_reported_downgrade_warns_but_does_not_regenerate(self):
        sent=[]
        def handler(request): sent.append(1); return sse_response('default')
        self.mock_http(handler)
        events=await self.stream('flex_only')
        self.assertEqual(len(sent),1)
        self.assertTrue(any('non-Flex tier' in e.get('routing_notice','') for e in events))
        self.assertFalse(next(e for e in events if e['type']=='usage')['routing']['flex_confirmed'])

    async def test_sse_error_with_usage_still_accounted_without_replay(self):
        sent=[]
        def handler(request):
            sent.append(1)
            chunk={'id':'billed-error','service_tier':'flex','provider':'OpenAI',
                   'usage':{'prompt_tokens':100,'completion_tokens':1,'cost':.001},
                   'error':{'code':503,'message':'generation interrupted'}}
            return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\n')
        self.mock_http(handler)
        events=await self.stream()
        self.assertEqual(len(sent),1)
        self.assertEqual(next(e for e in events if e['type']=='usage')['usage']['cost'],.001)
        self.assertEqual(events[-1]['type'],'error')

    async def test_nonstream_error_body_not_replayed(self):
        sent=[]
        def handler(request):
            sent.append(1)
            return httpx.Response(200,json={'error':{'message':'unknown error'}})
        self.mock_http(handler)
        with self.assertRaises(RuntimeError):
            await llm.complete(model='model',messages=[],pricing_mode='prefer_flex')
        self.assertEqual(len(sent),1)

    async def test_nonstream_summary_fallback_and_accounting(self):
        sent=[]; observed=[]
        def handler(request):
            sent.append(json.loads(request.content))
            if len(sent)==1: return httpx.Response(503,text='Flex unavailable')
            return httpx.Response(200,json={'id':'summary','provider':'OpenAI','service_tier':'default',
                 'usage':{'cost':.1},'choices':[{'message':{'content':'summary'}}]})
        self.mock_http(handler)
        async def callback(raw,**meta): observed.append(meta)
        result=await llm.complete(model='model',messages=[],pricing_mode='prefer_flex',usage_callback=callback)
        self.assertEqual(result,'summary')
        self.assertEqual(len(sent),2)
        self.assertEqual(observed[0]['routing']['reported_service_tier'],'default')
        self.assertFalse(observed[0]['routing']['flex_confirmed'])

    async def test_nonstream_strict_does_not_retry_effort_or_tier(self):
        sent=[]
        def handler(request): sent.append(1); return httpx.Response(429,text='Flex capacity')
        self.mock_http(handler)
        with self.assertRaises(httpx.HTTPStatusError):
            await llm.complete(model='model',messages=[],pricing_mode='flex_only')
        self.assertEqual(len(sent),1)

    async def test_accounting_uses_reported_not_requested_tier(self):
        accounting._PRICE_CACHE.clear()
        raw={'prompt_tokens':100,'completion_tokens':2,'cost':.0011,
             'completion_tokens_details':{'reasoning_tokens':0}}
        price=await accounting.pricing('model','OpenAI',100,raw,reported_tier='default')
        self.assertAlmostEqual(price['rates']['prompt'],.00001)
        raw['cost']=.00055
        price=await accounting.pricing('model','OpenAI',100,raw,reported_tier='flex')
        self.assertAlmostEqual(price['rates']['prompt'],.000005)
        usage={}
        accounting.record(usage,raw,model='model',price=price,routing={'reported_service_tier':'flex'})
        self.assertAlmostEqual(usage['cost'],.00055)  # not halved again!
        self.assertEqual(usage['recent_requests'][0]['routing']['reported_service_tier'],'flex')

    async def test_agent_and_summary_pass_independent_modes(self):
        s=make_session(); s.cfg.pricing_mode='prefer_flex'; s.cfg.compaction.summary_pricing_mode='flex_only'
        seen=[]
        async def stream(**kw):
            seen.append(kw); yield {'type':'text','text':'done'}
        async def complete(**kw): seen.append(kw); return 'summary'
        with patch('ctxagent.agent.llm.stream_chat',stream),patch('ctxagent.agent.llm.complete',complete):
            _=[e async for e in s.run_turn('test')]
            await s._complete(model='summary',messages=[])
        self.assertEqual(seen[0]['pricing_mode'],'prefer_flex')
        self.assertEqual(seen[1]['pricing_mode'],'flex_only')


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_metadata_only_endpoint_and_invalid_config(self):
        from ctxagent.server import app
        s=make_session()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            with patch.object(flex,'discover',AsyncMock(return_value={'available':True})) as discovery, \
                 patch.object(llm,'stream_chat') as generation:
                response=await client.post('/api/providers/flex',json={'model':'model','needs_tools':True,'refresh':True})
                self.assertEqual(response.status_code,200)
                self.assertTrue(discovery.call_args.kwargs['needs_tools'])
                generation.assert_not_called()
                response=await client.post('/api/providers/flex',json={'model':'model','policy':'invalid'})
                self.assertEqual(response.status_code,422)
            with patch('ctxagent.server.agent.get_session',return_value=s):
                response=await client.patch('/api/chats/test/config',json={'config':{'pricing_mode':'invalid'}})
                self.assertEqual(response.status_code,422)
                self.assertEqual(s.cfg.pricing_mode,'standard')
                response=await client.patch('/api/chats/test/config',json={'config':{'pricing_mode':'flex_only'}})
                self.assertEqual(response.status_code,200)
                self.assertEqual(s.cfg.pricing_mode,'flex_only')


if __name__=='__main__':
    unittest.main(verbosity=2)
