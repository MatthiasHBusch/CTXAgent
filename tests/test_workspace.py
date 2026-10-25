"""Offline document lifecycle, persistence, refresh and safe batching regressions."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent.config import SessionConfig, CompactionConfig
from ctxagent.segments import Conversation, K_SYSTEM, K_USER, K_TOOL_RESULT
from tests.test_compaction import pairing_ok
from ctxagent.tools import Sandbox, ToolError, ToolResult, REGISTRY
from ctxagent.workspace import DocumentWorkspace, WorkspaceError
from ctxagent.tool_batch import execute_batch, is_parallel_read
from ctxagent.agent import AgentSession


def call(n, path='a'):
    return {'id': path, 'type': 'function', 'function': {
        'name': n, 'arguments': json.dumps({'path': path})}}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'draft.py'
        self.path.write_text('def first():\n    return 1\n\ndef second():\n    return 2\n', encoding='utf-8')
        self.sb = Sandbox([str(self.root)])
        self.w = DocumentWorkspace()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_refresh_same_mtime_one_copy_and_close(self):
        doc_id = self.w.open(self.sb, str(self.path))
        original = self.path.stat()
        self.path.write_text('def first():\n    return 9\n\ndef second():\n    return 2\n', encoding='utf-8')
        import os
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.w.refresh(self.sb)
        self.assertIn('return 9', self.w.messages()[0]['content'])
        self.assertNotIn('return 1', self.w.messages()[0]['content'])
        self.assertEqual(self.w.open(self.sb, str(self.path), 'second'), doc_id)
        self.assertEqual(len(self.w.views), 1)
        self.assertIn('return 2', self.w.messages()[0]['content'])
        self.assertNotIn('return 9', self.w.messages()[0]['content'])
        self.w.close(doc_id)
        self.assertEqual(self.w.messages(), [])
        self.assertEqual(self.w.token_count(), 0)

    def test_failed_reopen_is_atomic_and_missing_has_no_stale_body(self):
        self.w.open(self.sb, str(self.path))
        with self.assertRaises(WorkspaceError):
            self.w.open(self.sb, str(self.path), 'nonexistent')
        self.assertIsNone(self.w.views[0]['select'])
        self.path.unlink()
        self.w.refresh(self.sb)
        self.assertEqual(self.w.views[0]['status'], 'unavailable')
        self.assertNotIn('return 1', self.w.messages()[0]['content'])
        self.path.write_text('recovered', encoding='utf-8')
        self.w.refresh(self.sb)
        self.assertEqual(self.w.views[0]['status'], 'ready')

    def test_limits_reject_and_block_growth(self):
        self.w.open(self.sb, str(self.path), max_tokens=200)
        self.path.write_text('large content ' * 1000, encoding='utf-8')
        self.w.refresh(self.sb)
        self.assertEqual(self.w.views[0]['status'], 'unavailable')
        self.assertEqual(self.w.views[0]['body'], '')
        with self.assertRaises(WorkspaceError):
            self.w.open(self.sb, str(self.path), max_tokens=200)
        for selectors in ([], 3, ['']):
            with self.assertRaises(WorkspaceError):
                self.w.open(self.sb, str(self.path), selectors)

    def test_persistence_is_spec_only_and_never_alters_frozen_history(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'system')
        c.new('user', K_USER, 'work on draft')
        c.freeze_prefix(2)
        before = json.dumps(c.to_messages())
        c.workspace.open(self.sb, str(self.path), 'first')
        encoded = c.dumps()
        self.assertNotIn('return 1', encoded)
        restored = Conversation.from_dict(json.loads(encoded))
        restored.workspace.refresh(self.sb)
        self.assertIn('return 1', restored.workspace.messages()[0]['content'])
        self.assertEqual(json.dumps(c.to_messages()), before)
        self.assertEqual(restored.frozen_upto, 2)
        self.assertEqual(Conversation.from_dict({'segments': []}).workspace.messages(), [])

    def test_permissions_rechecked(self):
        self.w.open(self.sb, str(self.path))
        self.w.refresh(Sandbox([]))
        self.assertEqual(self.w.views[0]['status'], 'unavailable')
        self.assertEqual(self.w.views[0]['body'], '')

    def test_pdf_extracted_full_view(self):
        import fitz
        path = self.root / 'test.pdf'
        doc = fitz.open()
        doc.new_page().insert_text((50, 50), 'workspace PDF text')
        doc.save(path)
        doc.close()
        self.w.open(self.sb, str(path))
        self.assertIn('workspace PDF text', self.w.messages()[0]['content'])

    def test_raw_line_range(self):
        self.w.open(self.sb, str(self.path), 'L1-2')
        self.assertIn('return 1', self.w.messages()[0]['content'])
        self.assertNotIn('return 2', self.w.messages()[0]['content'])


class BatchingTests(unittest.IsolatedAsyncioTestCase):
    async def test_overlap_bounded_order_and_mutation_barrier(self):
        active = 0
        peak = 0
        completed = []
        class Session:
            cancel = asyncio.Event()
            async def _exec_tool(self, name, args):
                nonlocal active, peak
                if name == 'edit_file':
                    assert active == 0
                    completed.append('edit')
                    return ToolResult('edited')
                if args['path'] == 'after':
                    assert 'edit' in completed
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.005 if args['path'] == 'a' else 0.001)
                active -= 1
                completed.append(args['path'])
                return ToolResult(args['path'])
        calls = [call('read_file', c) for c in 'abcdef'] + [call('edit_file', 'edit'), call('read_file', 'after')]
        events = [e async for e in execute_batch(Session(), calls)]
        self.assertEqual([e[1]['id'] for e in events if e[0] == 'result'], [c['id'] for c in calls])
        self.assertEqual(peak, 4)
        self.assertLess(completed.index('edit'), completed.index('after'))
        for name in ('run_command', 'document_open', 'recall', 'start_job', 'write_file'):
            self.assertFalse(is_parallel_read(name, {}))
        self.assertFalse(is_parallel_read('summarize_file', {'mode': 'llm'}))
        self.assertTrue(is_parallel_read('summarize_file', {'mode': 'static'}))

    async def test_bad_arguments_and_cancelled_serial_barrier(self):
        executed = []
        class Session:
            cancel = asyncio.Event()
            async def _exec_tool(self, name, args):
                executed.append(name)
                self.cancel.set()
                return ToolResult('ok')
        s = Session()
        calls = [call('run_command', 'a'), call('run_command', 'b')]
        results = [e[4] async for e in execute_batch(s, calls) if e[0] == 'result']
        self.assertEqual(executed, ['run_command'])
        self.assertIn('not executed', results[-1].text)
        s.cancel.clear()
        malformed = call('read_file')
        malformed['function']['arguments'] = '[]'
        events = [e async for e in execute_batch(s, [malformed])]
        self.assertIn('Bad arguments', events[-1][4].text)

    async def test_stable_document_slot_and_recorded_batch_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'draft.txt'
            path.write_text('current working draft', encoding='utf-8')
            cfg = SessionConfig(workdirs=[td], compaction=CompactionConfig(enabled=False))
            c = Conversation()
            c.new('system', K_SYSTEM, 'initial')
            c.new('assistant', 'assistant_text', 'older frozen work')
            c.freeze_prefix(2)
            c.new('user', K_USER, 'working-tail task')
            s = AgentSession('cache-slot', cfg, c)
            s.persist = lambda: None
            c.workspace.open(s.sandbox(), str(path))
            seen = []
            async def account(raw, **kw):
                s.usage.setdefault('recent_requests', []).append({'source': 'agent', 'usage': raw})
            s._account_usage = account
            async def stream(**kw):
                seen.append(kw['messages'])
                yield {'type': 'usage', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
                if len(seen) == 1:
                    yield {'type': 'tool_calls', 'tool_calls': [call('read_file', str(path))]}
                else:
                    yield {'type': 'text', 'text': 'done'}
                yield {'type': 'done', 'finish_reason': 'stop'}
            with patch('ctxagent.agent.llm.stream_chat', stream):
                _ = [e async for e in s.run_turn('')]
            self.assertIn('DOCUMENT OPERATION CATALOG', seen[0][2]['content'])
            self.assertIn('LIVE DOCUMENT WORKSPACE', seen[0][3]['content'])
            self.assertEqual(seen[0], seen[1][:len(seen[0])])
            self.assertEqual(seen[0][:2], c.to_messages()[:2])
            self.assertEqual([r['tool_calls'] for r in s.usage['recent_requests']], [1, 0])

    async def test_disconnect_cleans_read_tasks_before_unlocking(self):
        session = AgentSession('disconnect', SessionConfig(compaction=CompactionConfig(enabled=False)), Conversation())
        session.persist = lambda: None
        active = set()
        async def execute(name, args):
            active.add(args['path'])
            try:
                await asyncio.sleep(.001 if args['path'] == 'a' else 2)
                return ToolResult('read')
            finally:
                active.remove(args['path'])
        session._exec_tool = execute
        async def stream(**kw):
            yield {'type': 'tool_calls', 'tool_calls': [call('read_file', p) for p in 'abcd']}
            yield {'type': 'done', 'finish_reason': 'stop'}
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = session.run_turn('read independent files')
            async for event in events:
                if event['type'] == 'tool_result':
                    await events.aclose()
                    break
        self.assertEqual(active, set())
        self.assertFalse(session.turn_lock.locked())
        self.assertTrue(pairing_ok(session.conv))

    async def test_agent_live_view_refreshes_after_edit_and_close(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'draft.txt'
            path.write_text('version one', encoding='utf-8')
            cfg = SessionConfig(workdirs=[td], compaction=CompactionConfig(enabled=False))
            conv = Conversation()
            seen = []
            calls = [
                {'id': 'open', 'type': 'function', 'function': {'name': 'document_open',
                 'arguments': json.dumps({'path': str(path)})}},
                {'id': 'edit', 'type': 'function', 'function': {'name': 'write_file',
                 'arguments': json.dumps({'path': str(path), 'content': 'version two'})}},
            ]
            async def stream(**kw):
                seen.append(kw['messages'])
                if len(seen) <= 2:
                    yield {'type': 'tool_calls', 'tool_calls': [calls[len(seen)-1]]}
                elif len(seen) == 3:
                    doc_id = conv.workspace.views[0]['id']
                    yield {'type': 'tool_calls', 'tool_calls': [{'id': 'close', 'type': 'function',
                        'function': {'name': 'document_close', 'arguments': json.dumps({
                            'document_id': doc_id, 'summary': 'Updated draft to version two.'})}}]}
                else:
                    yield {'type': 'text', 'text': 'done'}
                yield {'type': 'done', 'finish_reason': 'stop'}
            s = AgentSession('workspace-test', cfg, conv)
            s.persist = lambda: None
            with patch('ctxagent.agent.llm.stream_chat', stream):
                events = [e async for e in s.run_turn('edit draft')]
            self.assertFalse([e for e in events if e['type'] == 'error'])
            self.assertIn('version one', next(m['content'] for m in seen[1] if 'LIVE DOCUMENT WORKSPACE' in str(m.get('content'))))
            self.assertIn('version two', next(m['content'] for m in seen[2] if 'LIVE DOCUMENT WORKSPACE' in str(m.get('content'))))
            self.assertNotIn('version one', next(m['content'] for m in seen[2] if 'LIVE DOCUMENT WORKSPACE' in str(m.get('content'))))
            self.assertFalse(any('LIVE DOCUMENT WORKSPACE' in str(m) for m in seen[3]))
            self.assertTrue(pairing_ok(conv))
            self.assertEqual(s.stats()['workspace_tokens'], 0)
            self.assertFalse(any('version one' in seg.body for seg in conv if seg.kind == K_TOOL_RESULT))
            self.assertTrue(any('Document handoff' in seg.body for seg in conv))
            self.assertIn('document_open', REGISTRY)


class WorkspaceAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_open_refresh_close_limits_and_busy_guard(self):
        import httpx
        from ctxagent.server import app
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'draft.txt'
            path.write_text('original file content', encoding='utf-8')
            s = AgentSession('api-workspace', SessionConfig(workdirs=[td]), Conversation())
            s.persist = lambda: None
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                with patch('ctxagent.server.agent.get_session', return_value=s):
                    r = await client.post('/api/chats/test/documents', json={'path': str(path)})
                    self.assertEqual(r.status_code, 200)
                    data = r.json()
                    doc_id = data['workspace']['views'][0]['id']
                    self.assertNotIn('original file content', json.dumps(data))
                    self.assertGreater(data['stats']['workspace_tokens'], 0)
                    prior = data['workspace']['views'][0]['version']
                    path.write_text('external file change', encoding='utf-8')
                    r = await client.get('/api/chats/test/documents')
                    self.assertNotEqual(r.json()['workspace']['views'][0]['version'], prior)
                    await s.turn_lock.acquire()
                    try:
                        self.assertEqual((await client.post('/api/chats/test/documents', json={'path': str(path)})).status_code, 409)
                        self.assertEqual((await client.post(f'/api/chats/test/documents/{doc_id}/close', json={})).status_code, 409)
                        self.assertEqual((await client.get('/api/chats/test/documents')).status_code, 200)
                    finally:
                        s.turn_lock.release()
                    r = await client.post('/api/chats/test/documents', json={'path': str(path), 'max_tokens': 0})
                    self.assertEqual(r.status_code, 422)
                    r = await client.post(f'/api/chats/test/documents/{doc_id}/close', json={'summary': 'Completed revision.'})
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.json()['workspace']['tokens'], 0)
                    self.assertTrue(any('Completed revision.' in v.body for v in s.conv))
                with patch('ctxagent.server.agent.get_session', return_value=None):
                    self.assertEqual((await client.get('/api/chats/no/documents')).status_code, 404)


if __name__ == '__main__':
    unittest.main()
