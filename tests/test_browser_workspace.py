"""Offline live browser lifecycle, prompt/cache, notes, summary and API regressions."""
import asyncio
import hashlib
import io
import json
import tempfile
import unittest
import httpx
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ctxagent import config, store, browsing, agent, browser_tools as bt, tokens
from ctxagent.config import SessionConfig
from ctxagent.segments import Conversation, K_USER, K_SYSTEM
from ctxagent.browser_workspace import BrowserWorkspace, WorkspaceError, notes, fingerprint, save_note, note_count, MAX_VIEWS
from ctxagent.tools import REGISTRY
from ctxagent.artifacts import save_png
from ctxagent.context_report import report
from ctxagent.delegation import read_only_name


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(config,'DATA_DIR',self.root),
            patch.object(config,'BLOB_DIR',self.root/'blobs'), patch.object(store,'DB_PATH',self.root/'chats.db')]
        for p in self.patches:p.start()
        store.init(); agent._SESSIONS.clear()
        cfg = SessionConfig(workdirs=[str(self.root)], enabled_tools=list(bt.BROWSER_TOOLS)+['document_open','document_close'])
        cfg.compaction.summary_model = 'fixture/summary'
        cfg.compaction.summary_pricing_mode = 'prefer_flex'
        cfg.compaction.summary_reasoning = 'off'
        self.cid = store.create_chat('research', cfg.to_dict())
        self.s = agent.AgentSession(self.cid,cfg,Conversation())
        agent._SESSIONS[self.cid] = self.s
        self.ctx = {'session':self.s}

    async def asyncTearDown(self):
        agent._SESSIONS.clear()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def page(self,text='Fixture fact: the speed is 42. Another fact is 17.',url='https://example.com/research',owner=None):
        data={'id':uuid.uuid4().hex,'owner':owner or self.cid,'url':url,'requested_url':url,
            'title':'Research source','kind':'html','text':text,'links':[{'text':'Further reading','url':'https://example.com/more'}],
            'sha256':hashlib.sha256(text.encode()).hexdigest(),'fetched_at':1234567890.0}
        (browsing._directory()/(data['id']+'.json')).write_text(json.dumps(data),encoding='utf-8')
        return data

    def attach(self,data,**kw):
        return self.s.conv.browser.attach(self.cid,data,**kw)

    async def call(self,name,**kw):
        result=await REGISTRY[name].fn(self.s.sandbox(),**kw,ctx=self.ctx)
        return json.loads(result.text)

    def image(self):
        from PIL import Image
        stream=io.BytesIO();Image.new('RGB',(64,64),'blue').save(stream,format='PNG')
        return save_png(stream.getvalue(),64,64,'Research screenshot','0'*64)

    async def test_open_receipt_is_small_and_live_once_not_serialized(self):
        data=self.page('ONLY_IN_LIVE_TEXT ' * 200)
        async def fetch(*a,**kw):return browsing._view(data,0,1,0),[]
        with patch.object(browsing,'open_page',side_effect=fetch):
            output=await self.call('browser_open',url=data['url'],max_tokens=6000)
        self.assertNotIn('ONLY_IN_LIVE_TEXT',json.dumps(output))
        self.assertIn('ONLY_IN_LIVE_TEXT',json.dumps(self.s.conv.browser.messages()))
        self.assertNotIn('ONLY_IN_LIVE_TEXT',self.s.conv.dumps())
        restored=Conversation.from_dict(self.s.conv.to_dict())
        restored.browser.refresh(self.cid)
        self.assertEqual(len(restored.browser.messages()),1)
        self.assertIn('ONLY_IN_LIVE_TEXT',json.dumps(restored.browser.messages()))
        self.assertGreater(self.s.stats()['browser_tokens'],0)
        self.assertGreater(report(self.s.conv,self.s.cfg)['context']['browser_workspace_tokens_estimate'],0)

    async def test_close_replaces_text_and_native_pixels_without_mutating_frozen_history(self):
        self.s.conv.new('user',K_USER,'frozen request');self.s.conv.freeze_prefix(2)
        before=json.dumps(self.s.conv.to_messages())
        data=self.page('VERY_LONG_FULL_PAGE ' * 100)
        tab=self.attach(data,images=[self.image()],max_tokens=6000)
        self.assertIn('image_ref',json.dumps(self.s.conv.browser.messages()))
        output=await self.call('browser_close',tab_id=tab,summary='Key source finding: 42.')
        self.assertEqual(output['closed'],[tab]);self.assertFalse(self.s.conv.browser.messages())
        self.assertEqual(self.s.conv.browser.token_count(),0)
        self.assertNotIn('VERY_LONG_FULL_PAGE',json.dumps(output))
        self.assertEqual(json.dumps(self.s.conv.to_messages()),before)
        note=notes(self.cid,output['handoffs'][0]['id'])
        self.assertEqual(note['text'],'Key source finding: 42.')
        self.assertEqual(note['page_id'],data['id'])
        self.assertNotIn('Key source finding',self.s.conv.dumps())
        self.assertEqual(Conversation.from_dict(self.s.conv.to_dict()).browser.views,[])

    async def test_full_summary_uses_summary_policy_and_accounting(self):
        data=self.page();tab=self.attach(data)
        seen=[]
        async def complete(**kw):
            seen.append(kw)
            await kw['usage_callback']({'prompt_tokens':30,'completion_tokens':10,'cost':.01},model=kw['model'])
            return 'Source states speed 42; not independently verified. https://example.com/research'
        with patch.object(agent.llm,'complete',side_effect=complete), patch.object(agent.accounting,'pricing',AsyncMock(return_value={'rates':{},'basis':'fixture'})):
            value=await self.call('browser_summarize',tab_id=tab)
        self.assertEqual(seen[0]['model'],'fixture/summary')
        self.assertEqual(seen[0]['pricing_mode'],'prefer_flex')
        self.assertEqual(seen[0]['reasoning_effort'],'off')
        self.assertEqual(seen[0]['provider_policy'],self.s.cfg.provider_policy)
        self.assertIn('untrusted',seen[0]['messages'][0]['content'])
        self.assertEqual(value['kind'],'llm_summary');self.assertEqual(value['coverage'],'page')
        self.assertEqual(self.s.usage['requests'],1)
        self.assertEqual(self.s.usage['recent_requests'][-1]['source'],'summary')
        self.assertAlmostEqual(self.s.usage['cost'],.01)
        self.assertTrue(self.s.conv.browser.views)

    async def test_selected_summary_is_explicit_and_long_page_does_not_silently_truncate(self):
        data=self.page('word ' * 80000);tab=self.attach(data,max_chars=2000)
        self.s._complete=AsyncMock(return_value='Source selection note')
        with self.assertRaisesRegex(WorkspaceError,'64000'):
            await self.call('browser_summarize',tab_id=tab)
        self.s._complete.assert_not_awaited()
        out=await self.call('browser_summarize',tab_id=tab,coverage='view')
        self.assertEqual(out['coverage'],'view');self.s._complete.assert_awaited_once()

    async def test_chunk_summary_covers_all_parts_and_reduces(self):
        data=self.page('unique start ' + 'word ' * 18000 + ' unique end');tab=self.attach(data,max_chars=2000)
        self.s._complete=AsyncMock(return_value='Source-based compact note')
        await self.call('browser_summarize',tab_id=tab)
        calls=self.s._complete.await_args_list
        self.assertGreater(len(calls),2)
        bodies=[c.kwargs['messages'][-1]['content'] for c in calls[:-1]]
        self.assertIn('unique start',bodies[0]);self.assertIn('unique end',bodies[-1])
        self.assertIn('CHUNK NOTES',calls[-1].kwargs['messages'][-1]['content'])

    async def test_reopen_cached_page_never_fetches_network(self):
        data=self.page('Cached page only')
        with patch('ctxagent.browsing._get',new=AsyncMock(side_effect=AssertionError('network forbidden'))) as network:
            opened=await bt.browser_open(None,page_id=data['id'],ctx=self.ctx)
        value=json.loads(opened.text)
        self.assertEqual(value['tab']['page_id'],data['id'])
        self.assertIn('Cached page only',self.s.conv.browser.messages()[0]['content'][0]['text'])
        self.assertNotIn('Cached page only',opened.text)
        network.assert_not_called()
        with self.assertRaisesRegex(WorkspaceError,'exactly one'):
            await bt.browser_open(None,url=data['url'],page_id=data['id'],ctx=self.ctx)
        with self.assertRaisesRegex(WorkspaceError,'immutable'):
            await bt.browser_open(None,page_id=data['id'],render_js=True,ctx=self.ctx)

    async def test_summary_honors_configured_output_limit_and_records_selection(self):
        data=self.page('First section. Selected section. Last section.')
        tab=self.attach(data,offset=15,max_chars=17)
        self.s.cfg.compaction.summary_max_tokens=100
        requests=[]
        async def complete(**kwargs):
            requests.append(kwargs)
            return 'Selected source claim, not independent verification.'
        with patch.object(self.s,'_complete',new=complete):
            result=await bt.browser_summarize(None,tab,max_tokens=800,coverage='view',ctx=self.ctx)
        saved=json.loads(result.text)
        self.assertEqual(requests[0]['max_tokens'],100)
        self.assertEqual(saved['selection'],{'offset':15,'end':32})
        self.assertIn('source_fetch_truncated=False',requests[0]['messages'][1]['content'])
        self.assertFalse(saved['source_fetch_truncated'])

    async def test_failed_cancelled_empty_or_overbudget_summary_preserves_tab(self):
        tab=self.attach(self.page())
        for value in (RuntimeError('provider rejected'), asyncio.CancelledError(), '', 'word '*5000):
            self.s._complete=AsyncMock(side_effect=value) if isinstance(value,BaseException) else AsyncMock(return_value=value)
            with self.assertRaises(BaseException):await self.call('browser_close',tab_id=tab)
            self.assertEqual(self.s.conv.browser.get(tab)['id'],tab)
            self.assertEqual(notes(self.cid)['total'],0)

    async def test_note_quotes_notebook_scope_and_bounded_index(self):
        tab=self.attach(self.page())
        value=await self.call('browser_note',tab_id=tab,note='A finding',quotation='speed is 42')
        self.assertEqual(notes(self.cid,value['id'])['quotation'],'speed is 42')
        with self.assertRaises(WorkspaceError):notes('other-chat',value['id'])
        with self.assertRaisesRegex(WorkspaceError,'Quotation'):
            await self.call('browser_note',tab_id=tab,note='Bad quote',quotation='not present')
        with self.assertRaises(WorkspaceError):await self.call('browser_note',tab_id=tab,note='word '*4000)
        index=notes(self.cid);self.assertNotIn('text',index['notes'][0]);self.assertNotIn('quotation',index['notes'][0])

    async def test_disk_tamper_owner_and_pixels_fail_closed(self):
        data=self.page();tab=self.attach(data,images=[self.image()],max_tokens=6000)
        with self.assertRaises(Exception):BrowserWorkspace().attach('other-chat',data)
        path=browsing._directory()/(data['id']+'.json')
        changed=json.loads(path.read_text());changed['text']='Injected changed text';path.write_text(json.dumps(changed))
        self.s.conv.browser.refresh(self.cid)
        self.assertEqual(self.s.conv.browser.get(tab)['status'],'unavailable')
        self.assertNotIn('Injected',json.dumps(self.s.conv.browser.messages()))
        path.write_text(json.dumps(data));image=self.s.conv.browser.get(tab)['images'][0]
        from ctxagent.artifacts import artifact_path
        artifact_path(image['id']).write_bytes(b'changed pixels')
        self.s.conv.browser.refresh(self.cid)
        self.assertIn('screenshot',self.s.conv.browser.get(tab)['error'])
        self.assertNotIn('image_ref',json.dumps(self.s.conv.browser.messages()))

    async def test_overbudget_view_and_shared_documents_preserve_existing_state(self):
        small=self.page();tab=self.attach(small);before=self.s.conv.browser.to_dict()
        with self.assertRaises(WorkspaceError):self.s.conv.browser.select(self.cid,tab,max_tokens=100,max_chars=40000)
        self.assertEqual(self.s.conv.browser.to_dict(),before)
        path=self.root/'large.txt';path.write_text('word '*2000)
        await REGISTRY['document_open'].fn(self.s.sandbox(),path=str(path),max_tokens=4000,ctx=self.ctx)
        self.s.cfg.context_window=13000;self.s.cfg.max_output_tokens=1000
        self.s.conv.browser.refresh(self.cid,max(0,bt.available(self.s)))
        self.assertEqual(self.s.conv.browser.get(tab)['status'],'unavailable')
        with self.assertRaises(WorkspaceError):await self.call('browser_view',tab_id=tab,max_tokens=4000)

    async def test_refresh_same_url_selection_and_render_live_only(self):
        data=self.page();tab=self.attach(data)
        fresh=self.page('Fresh rendered source')
        image=self.image();before=len(self.s.conv.segments)
        async def render(*a,**kw):
            self.assertTrue(kw['screenshot']);self.assertEqual(kw['scroll_y'],50)
            return browsing._view(fresh,0,1,0),[image]
        with patch.object(browsing,'open_page',side_effect=render):
            result=await self.call('browser_render',tab_id=tab,scroll_y=50)
        self.assertEqual(result['tab']['id'],tab);self.assertEqual(len(self.s.conv.browser.views),1)
        self.assertEqual(len(self.s.conv.segments),before)
        self.assertIn('Fresh rendered source',json.dumps(self.s.conv.browser.messages()))
        self.assertNotIn('Fresh rendered source',json.dumps(result))
        self.assertIn('image_ref',json.dumps(self.s.conv.browser.messages()))

    async def test_find_reselection_and_close_without_summary(self):
        text='prefix fact marker suffix';tab=self.attach(self.page(text))
        result=await self.call('browser_find',tab_id=tab,pattern='marker')
        self.assertTrue(result['matches'])
        await self.call('browser_view',tab_id=tab,offset=12,max_chars=6)
        self.assertIn('marker',json.dumps(self.s.conv.browser.messages()))
        self.s._complete=AsyncMock(side_effect=AssertionError('should not summarize'))
        result=await self.call('browser_close',summarize=False)
        self.assertEqual(result['closed'],[tab]);self.assertFalse(result['handoffs'])

    async def test_browser_tools_allowed_readonly_but_are_not_parallel_mutations(self):
        from ctxagent.tool_batch import is_parallel_read
        for name in bt.BROWSER_TOOLS:
            self.assertTrue(read_only_name(name),name)
            self.assertFalse(is_parallel_read(name,{}),name)

    async def test_api_scope_validation_running_guards_artifact_and_summary_handoff(self):
        import httpx
        from ctxagent.server import app
        tab=self.attach(self.page(),images=[self.image()],max_tokens=6000)
        self.s.persist();image=self.s.conv.browser.get(tab)['images'][0]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            listed=await client.get(f'/api/chats/{self.cid}/browser');self.assertEqual(listed.status_code,200)
            self.assertEqual(listed.json()['views'][0]['id'],tab)
            picture=await client.get(f"/api/chats/{self.cid}/artifacts/{image['id']}");self.assertEqual(picture.status_code,200)
            bad=await client.post(f'/api/chats/{self.cid}/browser/action',json={'tool':'run_command','arguments':{}})
            self.assertEqual(bad.status_code,422)
            bad=await client.post(f'/api/chats/{self.cid}/browser/action',json={'tool':'browser_close','arguments':{'extra':True}})
            self.assertEqual(bad.status_code,422)
            async with self.s.turn_lock:
                blocked=await client.post(f'/api/chats/{self.cid}/browser/action',json={'tool':'browser_close','arguments':{'summarize':False}})
                self.assertEqual(blocked.status_code,409)
            closed=await client.post(f'/api/chats/{self.cid}/browser/action',json={'tool':'browser_close','arguments':{'tab_id':tab,'summary':'42 is the source claim.'}})
            self.assertEqual(closed.status_code,200)
            self.assertEqual(self.s.conv.segments[-1].kind,'research_handoff')
            self.assertIn('42 is the source claim',self.s.conv.segments[-1].body)
            picture=await client.get(f"/api/chats/{self.cid}/artifacts/{image['id']}");self.assertEqual(picture.status_code,404)
            self.s.cfg.enabled_tools=[]
            disabled=await client.post(f'/api/chats/{self.cid}/browser/action',json={'tool':'browser_list'})
            self.assertEqual(disabled.status_code,403)

    async def test_close_all_without_summary_ignores_summary_output_cap(self):
        for i in range(MAX_VIEWS):
            self.attach(self.page('Fact '+str(i),url=f'https://example.org/{i}'))
        result=json.loads((await bt.browser_close(None,summarize=False,ctx=self.ctx)).text)
        self.assertEqual(len(result['closed']),MAX_VIEWS)
        self.assertEqual(result['handoffs'],[])
        self.assertEqual(self.s.conv.browser.token_count(),0)

    async def test_notebook_index_is_token_bounded_and_pagination_advances(self):
        long_url='https://example.org/?q='+'8f914fa350e5995037261211785b4f3a'*240
        data=self.page('Source fact',url=long_url)
        for i in range(3):
            save_note(self.cid,data,'Bounded note '+str(i))
        first=notes(self.cid,limit=50)
        self.assertLessEqual(tokens.count(json.dumps(first)),4000)
        self.assertGreater(first['next_offset'],0)
        self.assertTrue(first['notes'][0]['source_details_omitted'])
        self.assertNotIn('url',first['notes'][0])
        full=notes(self.cid,first['notes'][0]['id'])
        self.assertEqual(full['url'],long_url)
        second=notes(self.cid,offset=first['next_offset'],limit=50)
        self.assertGreater(second['next_offset'],first['next_offset'])

    async def test_delete_chat_removes_only_its_notebook(self):
        data=self.page()
        save_note(self.cid,data,'Owned note')
        other=store.create_chat('Other note owner',self.s.cfg.to_dict())
        save_note(other,data,'Other-owned note')
        store.delete_chat(self.cid)
        self.assertEqual(notes(self.cid)['total'],0)
        self.assertEqual(notes(other)['total'],1)

    async def test_ui_stop_cancels_summary_and_preserves_tab(self):
        from ctxagent import server
        tab=self.attach(self.page())
        started=asyncio.Event()
        async def pending(**kw):
            started.set()
            await asyncio.Event().wait()
        transport=httpx.ASGITransport(app=server.app)
        with patch.object(self.s,'_complete',new=pending):
            async with httpx.AsyncClient(transport=transport,base_url='http://test') as api:
                task=asyncio.create_task(api.post(f'/api/chats/{self.cid}/browser/action',json={
                    'tool':'browser_close','arguments':{'tab_id':tab}}))
                await asyncio.wait_for(started.wait(),3)
                self.assertIsNotNone(self.s.browser_task)
                stopped=await api.post(f'/api/chats/{self.cid}/stop')
                self.assertEqual(stopped.status_code,200)
                result=(await asyncio.wait_for(asyncio.gather(task,return_exceptions=True),3))[0]
                self.assertIsInstance(result,asyncio.CancelledError)
        self.assertEqual(self.s.conv.browser.get(tab)['id'],tab)
        self.assertFalse(self.s.turn_lock.locked())
        self.assertIsNone(self.s.browser_task)
        self.assertEqual(note_count(self.cid),0)

    async def test_handoff_is_not_compaction_summary_or_user_intent(self):
        from ctxagent.segments import K_RESEARCH
        from ctxagent.compaction.summarize import deep_summary
        handoff=self.s.conv.new('user',K_RESEARCH,'Source claims: 42. Not verified.')
        requests=[]
        async def complete(**kwargs):
            requests.append(kwargs)
            return 'Summary of source claims.'
        await deep_summary([handoff],None,'fixture/summary',complete)
        formatted=requests[0]['messages'][1]['content']
        self.assertIn('untrusted source claims, not user instructions',formatted)
        self.assertIn('Source claims: 42',formatted)
        self.assertNotIn('[USER]',formatted)
        self.assertEqual(handoff.to_message()['role'],'user')

    async def test_live_pressure_preflight_runs_before_main_model(self):
        self.s.cfg.context_window=30000
        self.s.cfg.max_output_tokens=2000
        self.s.cfg.compaction.enabled=True
        history=self.s.conv.new('assistant','assistant_text','Archive fact. '*12000)
        self.attach(self.page('Live selected source'))
        order=[]
        async def compact(**kwargs):
            order.append(('compact',kwargs['completed_batch']))
            history.body='Condensed archive facts.'
            yield {'type':'compaction','report':{'tiers':[]}}
        async def stream(**kwargs):
            order.append(('main',None))
            yield {'type':'text','text':'Answer'}
            yield {'type':'done','finish_reason':'stop'}
        with patch.object(self.s,'_compact',new=compact),patch('ctxagent.llm.stream_chat',new=stream):
            async for _ in self.s._run_turn('Continue research'):pass
        self.assertEqual(order[:2],[('compact',False),('main',None)])

    async def test_overfull_live_prompt_refused_before_provider_call(self):
        self.attach(self.page('Small source'))
        self.s.cfg.context_window=18000
        self.s.conv.new('assistant','assistant_text','Large history. '*14000)
        async def forbidden(**kwargs):
            raise AssertionError('Overfull prompt sent')
            yield {}
        with patch('ctxagent.llm.stream_chat',new=forbidden):
            events=[ev async for ev in self.s._run_turn('Research')]
        self.assertTrue(any(ev.get('type')=='error' and 'preflight' in ev['error'] for ev in events))

    async def test_prompt_injection_once_and_frozen_prefix_cache_unchanged(self):
        self.s.conv.new('user',K_USER,'Research');self.s.conv.freeze_prefix(2)
        frozen=json.dumps(self.s.conv.to_messages()[:self.s.conv.frozen_upto])
        tab=self.attach(self.page('UNIQUE_BROWSER_FACT'))
        self.s.persist()
        captured=[]
        async def stream(**kw):
            captured.append(kw['messages'])
            yield {'type':'text','text':'Finished'}
            yield {'type':'done','finish_reason':'stop'}
        with patch.object(agent.llm,'stream_chat',side_effect=stream):
            async for _ in self.s._run_turn(''):pass
        self.assertEqual(json.dumps(captured[0]).count('UNIQUE_BROWSER_FACT'),1)
        self.assertEqual(json.dumps(self.s.conv.to_messages()[:self.s.conv.frozen_upto]),frozen)
        self.s.conv.browser.close(tab)
        self.assertNotIn('UNIQUE_BROWSER_FACT',json.dumps(self.s.conv.to_messages()))


if __name__=='__main__':unittest.main()
