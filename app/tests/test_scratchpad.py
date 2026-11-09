"""The shared scratchpad: storage, agent edits, rebasing a stale user save, the tools and
the Ctrl+Enter message. Offline; the database lives in a temporary folder."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import blobs, config, scratchpad, store
from ctxagent.config import SessionConfig
from tests.test_queue import Scripted
from tests.test_productivity import make_session


class ScratchpadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for module, name, value in ((store, 'DB_PATH', root / 'test.db'), (config, 'DATA_DIR', root),
                                    (config, 'BLOB_DIR', root / 'blobs'), (blobs, 'BLOB_DIR', root / 'blobs')):
            p = patch.object(module, name, value)
            p.start()
            self.addCleanup(p.stop)
        store.init()
        self.cid = store.create_chat('pad', SessionConfig().to_dict())

    def test_empty_then_agent_modes(self):
        self.assertEqual(scratchpad.get(self.cid)['version'], 0)
        scratchpad.agent_write(self.cid, text='TODO\n- a')
        scratchpad.agent_write(self.cid, append='- b')
        state = scratchpad.agent_write(self.cid, edits=[{'old_string': '- a', 'new_string': '- a (done)'}])
        self.assertEqual(state['text'], 'TODO\n- a (done)\n- b')
        self.assertEqual((state['version'], state['updated_by']), (3, 'agent'))
        for bad in ({}, {'text': 'x', 'append': 'y'}, {'edits': [{'old_string': 'nope', 'new_string': ''}]},
                    {'edits': [{'old_string': '', 'new_string': 'x'}]}, {'append': '  '},
                    {'text': 'x' * (scratchpad.MAX_CHARS + 1)}):
            with self.assertRaises(scratchpad.ScratchpadError):
                scratchpad.agent_write(self.cid, **bad)
        self.assertEqual(scratchpad.get(self.cid)['version'], 3)

    def test_stale_user_save_keeps_the_agents_edits(self):
        base = scratchpad.save_user(self.cid, 'plan:\n1. read\n2. write', 0)['version']
        scratchpad.agent_write(self.cid, append='3. test (agent)')
        scratchpad.agent_write(self.cid, edits=[{'old_string': '1. read', 'new_string': '1. read (done)'}])
        # The user typed from `base` meanwhile: changed line 2.
        result = scratchpad.save_user(self.cid, 'plan:\n1. read\n2. write the summary', base)
        self.assertEqual(result['text'], 'plan:\n1. read (done)\n2. write the summary\n3. test (agent)')
        self.assertEqual(result['dropped_agent_edits'], 0)
        self.assertEqual(result['updated_by'], 'user')

    def test_conflicting_agent_edit_is_dropped_and_reported(self):
        base = scratchpad.save_user(self.cid, 'alpha beta', 0)['version']
        scratchpad.agent_write(self.cid, edits=[{'old_string': 'beta', 'new_string': 'BETA'}])
        scratchpad.agent_write(self.cid, text='rewritten by the agent')
        result = scratchpad.save_user(self.cid, 'alpha gamma', base)   # the user removed "beta"
        self.assertEqual(result['text'], 'alpha gamma')
        self.assertEqual(result['dropped_agent_edits'], 2)
        unchanged = scratchpad.save_user(self.cid, 'alpha gamma', result['version'])
        self.assertEqual(unchanged['version'], result['version'])

    def test_message_and_delete(self):
        self.assertIn('(the scratchpad is empty)', scratchpad.message(self.cid))
        scratchpad.save_user(self.cid, 'look at section 3', 0)
        text = scratchpad.message(self.cid)
        self.assertTrue(text.startswith(scratchpad.PREFIX))
        self.assertIn('<scratchpad>\nlook at section 3\n</scratchpad>', text)
        store.delete_chat(self.cid)
        self.assertEqual(scratchpad.get(self.cid)['version'], 0)

    async def test_tools_through_the_session(self):
        s = make_session()
        s.chat_id = self.cid
        s.cfg.enabled_tools = ['scratchpad_read', 'scratchpad_write']
        r = await s._exec_tool('scratchpad_write', {'append': 'from the agent'})
        self.assertFalse(r.error, r.text)
        r = await s._exec_tool('scratchpad_read', {})
        self.assertIn('<scratchpad>\nfrom the agent\n</scratchpad>', r.text)
        bad = await s._exec_tool('scratchpad_write', {'edits': [{'old_string': 'missing', 'new_string': 'x'}]})
        self.assertTrue(bad.error)
        self.assertIn('Nothing changed', bad.text)

    async def test_queued_scratchpad_uses_the_text_at_delivery(self):
        s = make_session()
        s.chat_id = self.cid
        scratchpad.save_user(self.cid, 'old text', 0)

        def user_edits_then_queues():
            s.enqueue('', 'scratchpad')
            scratchpad.save_user(self.cid, 'new text', 1)
        stream = Scripted([{'text': 'first', 'during': user_edits_then_queues}, {'text': 'second'}])
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [ev async for ev in s.run_turn('go')]
        delivered = [e for e in events if e['type'] == 'user_message']
        self.assertEqual(delivered[0]['kind'], 'scratchpad')
        self.assertIn('<scratchpad>\nnew text\n</scratchpad>', delivered[0]['segment']['body'])

    async def test_api(self):
        import httpx
        from ctxagent import server
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://t') as c:
            got = await c.get(f'/api/chats/{self.cid}/scratchpad')
            self.assertEqual(got.json()['version'], 0)
            saved = await c.put(f'/api/chats/{self.cid}/scratchpad', json={'text': 'hello', 'base_version': 0})
            self.assertEqual(saved.json()['version'], 1)
            too_long = await c.put(f'/api/chats/{self.cid}/scratchpad',
                                   json={'text': 'x' * (scratchpad.MAX_CHARS + 1), 'base_version': 1})
            self.assertEqual(too_long.status_code, 422)
            self.assertEqual((await c.get('/api/chats/nope/scratchpad')).status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
