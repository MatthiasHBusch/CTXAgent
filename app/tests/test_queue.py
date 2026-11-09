"""Messages sent while a turn runs: queued, delivered before the next request, handed back
when the turn stops. Offline: the model is a scripted stream."""
import unittest
from unittest.mock import AsyncMock, patch

from ctxagent.segments import K_USER
from ctxagent.tools import ToolResult
from tests.test_productivity import make_session


def call(cid='c1', name='read_file'):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}


class Scripted:
    """Each request takes the next scripted response; `during` runs while it streams."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, **kw):
        self.requests.append(kw['messages'])
        response = self.responses.pop(0)
        if response.get('during'):
            response['during']()
        if response.get('tools'):
            yield {'type': 'tool_calls', 'tool_calls': response['tools']}
        if response.get('text'):
            yield {'type': 'text', 'text': response['text']}
        yield {'type': 'done', 'finish_reason': 'tool_calls' if response.get('tools') else 'stop'}


def texts(messages):
    return [m['content'] if isinstance(m['content'], str) else ' '.join(
        b.get('text', '') for b in m['content'] if isinstance(b, dict)) for m in messages if m['role'] == 'user']


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, session, stream, text='start'):
        with patch('ctxagent.agent.llm.stream_chat', stream):
            return [ev async for ev in session.run_turn(text)]

    async def test_delivered_after_the_tool_batch(self):
        s = make_session()
        stream = Scripted([{'tools': [call()]}, {'text': 'done'}])

        async def tool(name, args):
            s.enqueue('also check the tests')   # the user types while the tool runs
            return ToolResult(text='file body')
        with patch.object(s, '_exec_tool', AsyncMock(side_effect=tool)):
            events = await self.run_turn(s, stream)
        delivered = [e for e in events if e['type'] == 'user_message']
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]['segment']['body'], 'also check the tests')
        second = stream.requests[1]
        roles = [m['role'] for m in second]
        self.assertEqual(roles[-2:], ['tool', 'user'])                 # right after the result
        self.assertEqual(texts(second)[-1], 'also check the tests')
        self.assertEqual(s.queue, [])
        self.assertEqual(events[-1]['type'], 'stats')
        self.assertFalse(any(e['type'] == 'queue_returned' for e in events))

    async def test_a_final_answer_with_a_queued_message_continues_the_turn(self):
        s = make_session()
        stream = Scripted([{'text': 'first answer', 'during': lambda: s.enqueue('one more thing')},
                           {'text': 'second answer'}])
        events = await self.run_turn(s, stream)
        finals = [e for e in events if e['type'] == 'final']
        self.assertEqual([f['text'] for f in finals], ['second answer'])
        self.assertEqual(len(stream.requests), 2)
        kinds = [(seg.kind, seg.body) for seg in s.conv.segments if seg.kind in (K_USER, 'assistant_text')]
        self.assertEqual(kinds, [(K_USER, 'start'), ('assistant_text', 'first answer'),
                                 (K_USER, 'one more thing'), ('assistant_text', 'second answer')])

    async def test_stop_hands_the_queue_back(self):
        s = make_session()

        def stop_and_queue():
            s.enqueue('do not lose me')
            s.cancel.set()
        stream = Scripted([{'tools': [call()], 'during': stop_and_queue}])
        with patch.object(s, '_exec_tool', AsyncMock(return_value=ToolResult(text='x'))):
            events = await self.run_turn(s, stream)
        returned = [e for e in events if e['type'] == 'queue_returned']
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]['reason'], 'cancelled')
        self.assertEqual([m['text'] for m in returned[0]['messages']], ['do not lose me'])
        self.assertEqual(s.queue, [])
        self.assertFalse(any(seg.body == 'do not lose me' for seg in s.conv.segments))

    async def test_unqueue_and_flags(self):
        s = make_session()
        item = s.enqueue('typo')
        self.assertTrue(s.unqueue(item['id']))
        self.assertFalse(s.unqueue(item['id']))
        self.assertFalse(s.turn_running)
        seen = []
        stream = Scripted([{'text': 'ok', 'during': lambda: seen.append(s.turn_running)}])
        await self.run_turn(s, stream)
        self.assertEqual(seen, [True])
        self.assertFalse(s.turn_running)

    async def test_endpoint_queues_only_while_a_turn_runs(self):
        import httpx
        from ctxagent import agent as agentmod, server
        s = make_session()
        with patch.object(agentmod, 'get_session', return_value=s), \
                patch.object(server, '_ordinary_chat', lambda cid: None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://t') as c:
                idle = await c.post('/api/chats/x/queue', json={'message': 'hi'})
                self.assertEqual(idle.status_code, 409)
                s.turn_running = True
                ok = await c.post('/api/chats/x/queue', json={'message': 'hi'})
                self.assertEqual(ok.status_code, 200)
                qid = ok.json()['queued']['id']
                self.assertEqual((await c.post('/api/chats/x/queue', json={'message': '  '})).status_code, 422)
                self.assertEqual((await c.delete(f'/api/chats/x/queue/{qid}')).status_code, 200)
                self.assertEqual((await c.delete(f'/api/chats/x/queue/{qid}')).status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
