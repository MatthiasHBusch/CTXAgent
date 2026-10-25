"""Offline delegation tests, including actual detached process/session roundtrips."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

_DATA=tempfile.TemporaryDirectory(prefix='ctxagent-agent-tests-')
os.environ.setdefault('CTXAGENT_DATA',_DATA.name)
from ctxagent import delegation as d, jobs, store, agent
from ctxagent.config import SessionConfig,CompactionConfig
from ctxagent.segments import Conversation,K_USER


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        store.init(); d.init()
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); (self.root/'input.txt').write_text('source evidence',encoding='utf-8')
        cfg=SessionConfig(workdirs=[str(self.root)],pricing_mode='prefer_flex',
            compaction=CompactionConfig(enabled=False,review_before_freeze=True))
        self.cid=store.create_chat('parent',cfg.to_dict())
        self.s=agent.get_session(self.cid)
        self.s.conv.new('user',K_USER,'parent history must not automatically leak')
        self.s.conv.workspace.open(self.s.sandbox(),str(self.root/'input.txt'))
        self.s.persist()
        self.ids=[]

    def tearDown(self):
        for identifier in self.ids:
            try:
                r=d._row(identifier)
                if r['job']:
                    jobs.cancel(r['job'],r['root'])
                    deadline=time.monotonic()+8
                    while jobs.status(r['job'],r['root'])['state'] not in jobs.TERMINAL and time.monotonic()<deadline:
                        time.sleep(.05)
                d.finish(identifier,'','test cleanup','cancelled')
            except ValueError:
                pass
        agent.drop_session(self.cid)

    def stub(self,*args,**kwargs):
        with patch.object(d,'_launch'):
            data=d.start(self.s,*args,**kwargs)
        self.ids.append(data['id'])
        return data

    async def test_settings_workspace_and_handoff_are_independent_deep_copies(self):
        original=copy.deepcopy(self.s.cfg.to_dict())
        a=self.stub('review',context='explicit context')
        row=store.get_chat(a['child'])
        self.assertEqual(row['config'],original)
        child=agent.get_session(a['child'])
        self.assertIsNot(child.cfg.compaction,self.s.cfg.compaction)
        child.cfg.workdirs.append('different')
        child.cfg.compaction.summary_reasoning='off'
        self.assertEqual(self.s.cfg.to_dict(),original)
        self.assertEqual(len(child.conv.workspace.views),1)
        child.conv.workspace.views.clear()
        self.assertEqual(len(self.s.conv.workspace.views),1)
        text=json.dumps(row['conversation'])
        self.assertIn('explicit context',text)
        self.assertNotIn('parent history must not automatically leak',text)
        self.assertEqual(child.usage['requests'],0)

    async def test_partial_overrides_and_exact_segment_handoff(self):
        seg=self.s.conv.segments[-1]
        a=self.stub('review',overrides={'model':'another','compaction':{'hot_zone_tokens':12000}},segment_ids=[seg.id],inherit_documents=False)
        row=store.get_chat(a['child'])
        self.assertEqual(row['config']['model'],'another')
        self.assertEqual(row['config']['pricing_mode'],'prefer_flex')
        self.assertTrue(row['config']['compaction']['review_before_freeze'])
        self.assertEqual(row['config']['compaction']['hot_zone_tokens'],12000)
        self.assertIn('parent history must not automatically leak',json.dumps(row['conversation']))
        self.assertEqual(agent.get_session(a['child']).conv.workspace.views,[])
        for bad in ({'unknown':1},{'compaction':{'unknown':1}},{'agent_max_parallel':99}):
            with self.assertRaises(ValueError): self.stub('bad',overrides=bad)
        with self.assertRaises(ValueError): self.stub('bad',segment_ids=['missing'])

    async def test_runtime_readonly_guard_and_descendant_cannot_remove_it(self):
        a=self.stub('review',read_only=True)
        child=agent.get_session(a['child'])
        r=await child._exec_tool('write_file',{'path':'input.txt','content':'bad'})
        self.assertTrue(r.error)
        self.assertNotIn('write_file',child.tool_names())
        from ctxagent import document_catalog
        self.assertNotIn('document_json_update',document_catalog.operation_names(child))
        r=await child._exec_tool('read_file',{'path':'input.txt'})
        self.assertFalse(r.error)
        self.assertEqual((self.root/'input.txt').read_text(),'source evidence')
        self.assertTrue(d.claim(a['id']))
        with patch.object(d,'_launch'):
            b=d.start(child,'nested',read_only=False,overrides={'agent_max_depth':4})
        self.ids.append(b['id'])
        self.assertTrue(b['read_only'])
        grandchild=agent.get_session(b['child'])
        with self.assertRaises(ValueError): d.start(grandchild,'depth bypass',overrides={'agent_max_depth':4})

    async def test_ownership_status_result_cursor_and_terminal_fencing(self):
        a=self.stub('review')
        with self.assertRaises(PermissionError): d.status(a['id'],'stranger')
        with self.assertRaises(PermissionError): d.result(a['id'],'stranger')
        d.finish(a['id'],'x'*45000)
        result=d.result(a['id'],self.cid,max_chars=40000)
        self.assertFalse(result['at_end']); self.assertEqual(len(result['text']),40000)
        self.assertTrue(d.result(a['id'],self.cid,offset=40000)['at_end'])
        d.finish(a['id'],'','late supervisor failure','failed')
        self.assertEqual(d.status(a['id'],self.cid)['state'],'succeeded')
        self.assertEqual(d.result(a['id'],self.cid)['text'],'x'*12000)
        self.assertEqual(self.s.usage['requests'],0)

    async def test_atomic_parallel_slot_claims_and_recovery(self):
        self.s.cfg.agent_max_parallel=2
        tasks=[self.stub('slot') for _ in range(5)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            successes=list(pool.map(d.claim,[a['id'] for a in tasks]))
        self.assertEqual(sum(successes),2)
        first=next(a for a,ok in zip(tasks,successes) if ok)
        d.finish(first['id'],'done')
        queued=next(a for a,ok in zip(tasks,successes) if not ok)
        self.assertTrue(d.claim(queued['id']))
        self.assertFalse(d.claim(queued['id']))

    async def test_supervisor_loss_reclaims_slot_without_replay(self):
        a=self.stub('lost')
        with store._conn() as c:
            c.execute('UPDATE agent_tasks SET job=? WHERE id=?',('f'*32,a['id']))
        with patch.object(jobs,'status',return_value={'state':'lost'}):
            status=d.status(a['id'],self.cid)
        self.assertEqual(status['state'],'lost')
        with self.assertRaises(ValueError): d.send(a['id'],self.cid,'retry')

    async def test_cancel_wins_terminal_race_and_followup_retains_context(self):
        a=self.stub('first')
        d.cancel(a['id'],self.cid)
        d.finish(a['id'],'done')
        self.assertEqual(d.status(a['id'],self.cid)['state'],'cancelled')
        with patch.object(d,'_launch'):
            sent=d.send(a['id'],self.cid,'follow up',{'compaction':{'summary_reasoning':'off'}})
        self.assertEqual(sent['child'],a['child'])
        self.assertEqual(sent['state'],'queued')
        row=store.get_chat(a['child'])
        self.assertEqual(row['config']['compaction']['summary_reasoning'],'off')
        self.assertEqual(len(agent.get_session(a['child']).conv.workspace.views),1)
        with self.assertRaises(ValueError): d.send(a['id'],self.cid,'race')

    def fixture_launch(self,identifier,timeout,root):
        script=str(Path(__file__).with_name('agent_fixture_worker.py').resolve())
        quote=lambda s:"'"+s.replace("'","''")+"'"
        command=('& '+quote(sys.executable)+' -u '+quote(script)+' '+quote(identifier)) if sys.platform=='win32' else ' '.join(map(shlex.quote,[sys.executable,'-u',script,identifier]))
        def publish(jid):
            with store._conn() as c: c.execute('UPDATE agent_tasks SET job=? WHERE id=?',(jid,identifier))
        jobs.start(command,Path(script).parent.parent,root,timeout,publish=publish,preserve_managed_children=True)

    async def wait(self,identifier,limit=20):
        deadline=asyncio.get_running_loop().time()+limit
        while asyncio.get_running_loop().time()<deadline:
            r=await asyncio.to_thread(d.status,identifier,self.cid)
            if r['state'] in d.TERMINAL:return r
            await asyncio.sleep(.1)
        self.fail('Worker did not complete in time: '+str(d.status(identifier,self.cid)))

    async def test_actual_detached_workers_tools_persistence_and_parent_stop(self):
        with patch.object(d,'_launch',side_effect=self.fixture_launch):
            tasks=[d.start(self.s,'review independent '+str(i),read_only=True) for i in range(2)]
        self.ids.extend(a['id'] for a in tasks)
        self.s.cancel.set()  # parent Stop must not cancel children
        results=await asyncio.gather(*(self.wait(a['id']) for a in tasks))
        self.assertTrue(all(r['state']=='succeeded' for r in results),results)
        for a in tasks:
            self.assertIn('source evidence',d.result(a['id'],self.cid)['text'])
            row=store.get_chat(a['child'])
            segs=row['conversation']['segments']
            self.assertTrue(any(s['kind']=='tool_result' for s in segs))
            self.assertEqual(row['config']['compaction'],self.s.cfg.compaction.to_dict())
            self.assertEqual(row['usage']['requests'],0) # fixture never reports paid usage

    async def test_actual_supervisor_cancel_and_timeout(self):
        with patch.object(d,'_launch',side_effect=self.fixture_launch):
            cancelled=d.start(self.s,'BLOCK forever',timeout=20)
            timed=d.start(self.s,'BLOCK timeout',timeout=3)
        self.ids.extend([cancelled['id'],timed['id']])
        d.cancel(cancelled['id'],self.cid)
        self.assertEqual((await self.wait(cancelled['id']))['state'],'cancelled')
        self.assertEqual((await self.wait(timed['id']))['state'],'timed_out')

    async def test_nested_managed_agent_survives_parent_worker_exit(self):
        with patch.object(d,'_launch',side_effect=self.fixture_launch):
            parent=d.start(self.s,'NEST a child',timeout=20,read_only=True)
        self.ids.append(parent['id'])
        self.assertEqual((await self.wait(parent['id']))['state'],'succeeded')
        deadline=asyncio.get_running_loop().time()+8
        descendants=[]
        while asyncio.get_running_loop().time()<deadline:
            descendants=[a for a in d.list_agents(self.cid) if a['parent']==parent['child']]
            if descendants and jobs.status(parent['job'],self.cid)['state'] in jobs.TERMINAL: break
            await asyncio.sleep(.1)
        self.assertEqual(len(descendants),1)
        child=descendants[0]; self.ids.append(child['id'])
        self.assertNotIn(d.status(child['id'],self.cid)['state'],d.TERMINAL)
        d.cancel(parent['id'],self.cid,descendants=True)
        self.assertEqual((await self.wait(child['id']))['state'],'cancelled')

    async def test_api_view_is_worker_owned_and_stale_cache_is_reloaded(self):
        import httpx
        from ctxagent.server import app
        a=self.stub('review')
        child=agent.get_session(a['child'])
        conv=Conversation.from_dict(store.get_chat(a['child'])['conversation'])
        conv.new('user',K_USER,'fresh checkpoint')
        store.save_chat(a['child'],conversation=conv.to_dict())
        self.assertIn('fresh checkpoint',json.dumps(agent.get_session(a['child']).conv.to_dict()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            r=await c.get('/api/chats/'+a['child']); self.assertTrue(r.json()['running'])
            for path,body in [('/send',{'message':'race'}),('/config',{'config':{'model':'race'}}),('/compact',{})]:
                r=await c.request('PATCH' if path=='/config' else 'POST','/api/chats/'+a['child']+path,json=body)
                self.assertEqual(r.status_code,409,r.text)
            r=await c.get('/api/chats/'+self.cid+'/agents')
            self.assertEqual(r.json()['agents'][0]['id'],a['id'])
            r=await c.get('/api/chats/stranger/agents/'+a['id']); self.assertEqual(r.status_code,404)

    async def test_descendant_costs_are_counted_once_and_survive_chat_deletion(self):
        a=self.stub('first')
        d.claim(a['id'])
        child=agent.get_session(a['child'])
        with patch.object(d,'_launch'):
            b=d.start(child,'nested')
        self.ids.append(b['id'])
        store.save_chat(a['child'],usage={'requests':2,'cost':.1,'missing_cost_requests':1})
        store.save_chat(b['child'],usage={'requests':3,'cost':.2,'missing_cost_requests':0})
        total=d.cost_summary(self.cid)
        self.assertAlmostEqual(total['cost'],.3)
        self.assertEqual(total['requests'],5)
        self.assertEqual(total['missing_cost_requests'],1)
        self.assertEqual(total['agents'],2)
        self.assertAlmostEqual(d.cost_summary(a['child'])['cost'],.2)
        self.assertEqual(self.s.usage['cost'],0) # parent ledger not mutated
        d.archive_chat_usage(b['child']); store.delete_chat(b['child'])
        self.assertAlmostEqual(d.cost_summary(self.cid)['cost'],.3)
        self.assertAlmostEqual(self.s.stats()['agents_cost']['cost'],.3)

    async def test_override_types_budget_and_queue_preflight(self):
        for overrides in ({'enabled_tools':'read_file'},{'workdirs':'C:/bad'}, {'context_window':20000,'max_output_tokens':19000}):
            with self.assertRaises(ValueError): self.stub('review',overrides=overrides)
        with patch.object(d,'MAX_PENDING_PER_ROOT',1):
            self.stub('first')
            with self.assertRaises(ValueError): self.stub('full queue')

    async def test_builtin_writes_are_serialized_across_sessions(self):
        from ctxagent import write_coordination
        active=0; peak=0
        async def write():
            nonlocal active,peak
            active+=1; peak=max(peak,active)
            await asyncio.sleep(.03)
            active-=1
        await asyncio.gather(*(write_coordination.execute('write_file',{},write) for _ in range(3)))
        self.assertEqual(peak,1)


if __name__=='__main__': unittest.main(verbosity=2)
