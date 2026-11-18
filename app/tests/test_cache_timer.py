"""The cache timer beside the send button (2026-11-11). AgentSession.cache_clock says when the
newest request reached the provider and how long its cache lives; a click queues a refresh,
which is sent KEEPALIVE_S after the request or keep-alive before it and keeps the cache that
much longer. Several can be queued, and taken back."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ctxagent import agent as agentmod, anthropic_api, config, llm
from ctxagent.agent import AgentSession
from ctxagent.config import SessionConfig
from ctxagent.segments import Conversation

DIRECT = 'claude-opus-5-5'
READ = {'input_tokens': 3, 'cache_read_input_tokens': 100_000, 'cache_creation_input_tokens': 0, 'output_tokens': 0}
WROTE = {'input_tokens': 3, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 100_000, 'output_tokens': 0}
STEP, TTL = 0.2, 0.4          # KEEPALIVE_S and CACHE_TTL_S in these tests


class CacheTimer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in (patch.object(config, 'DATA_DIR', Path(self.tmp.name)),
                  patch.object(agentmod, 'KEEPALIVE_S', STEP),
                  patch.object(agentmod, 'CACHE_TTL_S', TTL),
                  patch('ctxagent.agent.accounting.pricing', AsyncMock(return_value={'rates': {}, 'basis': 'test'}))):
            p.start()
            self.addCleanup(p.stop)
        llm.KEEPALIVE_REFUSED.clear()
        self.addCleanup(llm.KEEPALIVE_REFUSED.clear)

    def session(self, model=DIRECT):
        s = AgentSession('timer', SessionConfig(model=model, workdirs=[self.tmp.name]), Conversation())
        s.persist = lambda: None
        return s

    def warm(self, s, age=0.0):
        """What a request of a turn leaves: the captured request and when it was sent."""
        s._warm = {'route': 'anthropic', 'kwargs': {'model': s.cfg.model}, 'started': time.time() - age}
        return s._warm

    def keep(self, usage=READ):
        return patch.object(llm, 'keep_warm', AsyncMock(return_value={
            'usage': anthropic_api.usage_payload(DIRECT, dict(usage)), 'provider': 'Anthropic', 'generation_id': None}))

    # ---- the clock ----------------------------------------------------------------------

    async def test_no_timer_before_the_first_request(self):
        s = self.session()
        clock = s.cache_clock()
        self.assertIsNone(clock['warm_at'])
        self.assertFalse(clock['refreshable'])
        with self.assertRaises(ValueError):
            s.queue_refresh()

    async def test_a_turn_starts_the_timer_and_reports_it(self):
        s = self.session()

        async def stream(**kw):
            kw['capture'].update({'route': 'anthropic', 'kwargs': {'model': kw['model']}, 'started': time.time()})
            yield {'type': 'text', 'text': 'hi'}
            yield {'type': 'usage', 'usage': anthropic_api.usage_payload(DIRECT, dict(READ))}
            yield {'type': 'done', 'finish_reason': 'stop'}
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [ev async for ev in s.run_turn('hello')]
        clock = next(ev for ev in events if ev['type'] == 'usage')['cache']
        self.assertEqual(clock['warm_at'], s._warm['started'])
        self.assertEqual((clock['ttl'], clock['step'], clock['queued']), (TTL, STEP, 0))
        self.assertIsNone(s._capture)                        # the turn's own keep-alive ended with it
        self.assertTrue(s.cache_clock()['refreshable'])      # the request to repeat stays

    async def test_after_a_restart_the_time_is_shown_but_nothing_can_be_refreshed(self):
        s = self.session()
        s.conv.cache_state = {'sent_at': time.time()}
        clock = s.cache_clock()
        self.assertEqual(clock['warm_at'], s.conv.cache_state['sent_at'])
        self.assertFalse(clock['refreshable'])
        self.assertIn('since the app started', clock['why_not'])
        with self.assertRaises(ValueError):
            s.queue_refresh()
        other = self.session('openai/gpt-6.1-sol')            # no 5-minute cache to count down
        other.conv.cache_state = {'sent_at': time.time()}
        self.assertIsNone(other.cache_clock()['warm_at'])

    # ---- the queue ----------------------------------------------------------------------

    async def test_queued_refreshes_are_sent_one_after_the_other(self):
        s = self.session()
        warm = self.warm(s)
        with self.keep() as keep:
            s.queue_refresh()
            clock = s.queue_refresh()
            self.assertEqual((clock['queued'], clock['refreshable']), (2, True))
            self.assertEqual(keep.await_count, 0)             # not at once: shortly before the end
            await asyncio.sleep(STEP * 1.5)
            self.assertEqual((keep.await_count, s.refreshes), (1, 1))
            self.assertGreater(s.cache_clock()['warm_at'], warm['started'])
            await asyncio.sleep(STEP * 3)
            self.assertEqual((keep.await_count, s.refreshes), (2, 0))    # and none once the queue is empty
        self.assertIs(keep.await_args.args[0], warm)          # the last request, again
        self.assertEqual(s.usage['by_source']['keepalive']['requests'], 2)

    async def test_a_newer_request_restarts_the_wait(self):
        s = self.session()
        self.warm(s)
        with self.keep() as keep:
            s.queue_refresh()
            await asyncio.sleep(STEP * 0.6)
            newer = self.warm(s)                              # a request of a running turn
            await asyncio.sleep(STEP * 0.75)                  # the first one's time has passed
            self.assertEqual((keep.await_count, s.refreshes), (0, 1))
            await asyncio.sleep(STEP * 0.75)
            self.assertEqual((keep.await_count, s.refreshes), (1, 0))
            self.assertIs(keep.await_args.args[0], newer)

    async def test_a_running_tool_batch_sends_its_own_keepalive(self):
        s = self.session()
        self.warm(s)
        s._keeper = asyncio.create_task(asyncio.sleep(30))
        self.addCleanup(s._keeper.cancel)
        with self.keep() as keep:
            s.queue_refresh()
            await asyncio.sleep(STEP * 1.75)
            self.assertEqual((keep.await_count, s.refreshes), (0, 1))

    async def test_refreshes_can_be_taken_back_and_the_queue_is_capped(self):
        s = self.session()
        self.warm(s)
        with self.keep() as keep:
            for _ in range(agentmod.REFRESH_QUEUE_MAX + 5):
                clock = s.queue_refresh()
            self.assertEqual(clock['queued'], agentmod.REFRESH_QUEUE_MAX)
            self.assertEqual(s.queue_refresh(-1)['queued'], agentmod.REFRESH_QUEUE_MAX - 1)
            self.assertEqual(s.queue_refresh(-100)['queued'], 0)
            await asyncio.sleep(STEP * 1.5)
            self.assertEqual(keep.await_count, 0)

    async def test_a_refresh_that_missed_the_cache_drops_the_rest(self):
        s = self.session()
        self.warm(s)
        with self.keep(WROTE) as keep:
            for _ in range(3):
                s.queue_refresh()
            await asyncio.sleep(STEP * 2.5)
            self.assertEqual((keep.await_count, s.refreshes), (1, 0))
        self.assertIn('cache not hit', s.cache_clock()['note'])
        self.warm(s)
        with self.keep():
            self.assertIsNone(s.queue_refresh()['note'])      # a new click starts over
            s.queue_refresh(-1)

    async def test_a_failed_refresh_drops_the_rest_and_says_why(self):
        s = self.session()
        self.warm(s)
        with patch.object(llm, 'keep_warm', AsyncMock(side_effect=llm.KeepAliveRefused('HTTP 400: no'))) as keep:
            s.queue_refresh()
            s.queue_refresh()
            await asyncio.sleep(STEP * 1.5)
            self.assertEqual((keep.await_count, s.refreshes), (1, 0))
        clock = s.cache_clock()
        self.assertIn('HTTP 400', clock['note'])
        self.assertFalse(clock['refreshable'])                # refused: none more on this route
        with self.assertRaises(ValueError):
            s.queue_refresh()

    async def test_an_expired_cache_takes_no_refresh(self):
        s = self.session()
        self.warm(s, age=TTL * 2)
        with self.assertRaises(ValueError) as caught:
            s.queue_refresh()
        self.assertIn('expired', str(caught.exception))
        self.assertEqual(s.queue_refresh(-1)['queued'], 0)    # taking back is always allowed

    # ---- the routes ---------------------------------------------------------------------

    async def test_routes(self):
        import httpx
        from ctxagent import server
        s = self.session()
        with patch.object(agentmod, 'get_session', return_value=s), \
                patch.object(server, '_ordinary_chat', lambda cid: None), self.keep():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://t') as c:
                self.assertEqual((await c.post('/api/chats/x/cache/refresh', json={})).status_code, 409)
                self.warm(s)
                added = await c.post('/api/chats/x/cache/refresh', json={})
                self.assertEqual((added.status_code, added.json()['queued']), (200, 1))
                seen = (await c.get('/api/chats/x/cache')).json()
                self.assertEqual(seen['queued'], 1)
                self.assertIn('usage', seen)
                back = await c.post('/api/chats/x/cache/refresh', json={'delta': -1})
                self.assertEqual(back.json()['queued'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
