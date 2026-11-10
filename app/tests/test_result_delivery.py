"""Bounded result delivery (initial_result_max_tokens), without model calls. (The review
before freezing that this file also covered went with the frozen prefix on 2026-11-10.)"""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from ctxagent import blobs, config, result_delivery
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_TOOL_RESULT
from tests.test_productivity import make_session


def settings(**kw):
    d = dict(rewrite_interval_batches=5, deep_enabled=False)
    d.update(kw)
    return CompactionConfig(**d)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for target, attr, value in [(config, 'DATA_DIR', root), (config, 'BLOB_DIR', root / 'blobs'),
                                    (blobs, 'BLOB_DIR', root / 'blobs')]:
            p = patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.c = Conversation()
        self.full = '\n'.join(f'{i} data scientifically important context details ' * 12 for i in range(1500))
        self.seg = self.c.new('tool', K_TOOL_RESULT, self.full, tool_name='read_file')
        self.cc = settings(initial_result_max_tokens=4000)

    async def test_archive_cap_reload_and_exact_recall(self):
        freed = result_delivery.prepare(self.seg, self.cc, 'delivery')
        self.assertGreater(freed, 0)
        self.assertLessEqual(self.seg.tokens(), 4000)
        self.assertIn('INCOMPLETE', self.seg.body)
        self.assertEqual(blobs.get_blob(self.seg.blob_id), self.full)
        c = Conversation.from_dict(self.c.to_dict())
        session = make_session(c)
        recalled = await session._exec_tool('recall', {'segment_id': self.seg.id})
        self.assertEqual(recalled.text.split('\n', 1)[1], self.full)
        self.assertTrue(recalled.pinned)
        self.assertIn('INCOMPLETE', c.by_id(self.seg.id).body)

    async def test_agent_delivers_cap_before_next_request_and_does_not_repeat_on_storage_failure(self):
        from ctxagent.tools import ToolResult
        from unittest.mock import AsyncMock
        for fail in (False, True):
            with self.subTest(storage_failure=fail):
                session = make_session()
                session.cfg.compaction = session.engine.cfg = self.cc
                captured, calls = [], 0
                async def stream(**kw):
                    nonlocal calls
                    calls += 1
                    captured.append(copy.deepcopy(kw['messages']))
                    if calls == 1:
                        yield {'type': 'tool_calls', 'tool_calls': [
                            {'id': 'first', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}
                    else:
                        yield {'type': 'text', 'text': 'done'}
                    yield {'type': 'done', 'finish_reason': 'stop'}
                execute = AsyncMock(return_value=ToolResult(text=self.full))
                real_put = result_delivery.put_blob
                with patch('ctxagent.agent.llm.stream_chat', stream), patch.object(session, '_exec_tool', execute), patch(
                        'ctxagent.result_delivery.put_blob', side_effect=OSError('storage failure') if fail else real_put):
                    events = [ev async for ev in session.run_turn('test bounded delivery')]
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(calls, 2)
                delivered = next(m['content'] for m in captured[1] if m['role'] == 'tool')
                if fail:
                    self.assertEqual(delivered, self.full)
                else:
                    self.assertIn('INCOMPLETE', delivered)
                    self.assertLessEqual(next(ev['tokens'] for ev in events if ev['type'] == 'tool_result'), 4000)

    async def test_cap_disabled_and_exemptions(self):
        for changes, error in [({'pinned': True}, False), ({'tool_name': 'recall'}, False),
                               ({'images': [{'id': 'visual'}]}, False), ({}, True)]:
            seg = self.c.new('tool', K_TOOL_RESULT, self.full, **changes)
            self.assertEqual(result_delivery.prepare(seg, self.cc, 'delivery', error=error), 0)
            self.assertEqual(seg.body, self.full)
        self.cc.initial_result_max_tokens = 0
        self.assertEqual(result_delivery.prepare(self.seg, self.cc, 'delivery'), 0)
        self.cc.initial_result_max_tokens = 4000
        self.cc.enabled = False
        self.assertEqual(result_delivery.prepare(self.seg, self.cc, 'delivery'), 0)

    async def test_storage_failure_leaves_full_evidence(self):
        with patch('ctxagent.result_delivery.put_blob', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                result_delivery.prepare(self.seg, self.cc, 'delivery')
        self.assertEqual(self.seg.body, self.full)
        self.assertIsNone(self.seg.blob_id)

    def test_config_validation_roundtrip(self):
        cc = CompactionConfig.from_dict(self.cc.to_dict())
        self.assertEqual(cc.initial_result_max_tokens, 4000)
        for bad in (-1, 100, 100001, float('nan')):
            with self.assertRaises((ValueError, OverflowError)):
                CompactionConfig.from_dict({'initial_result_max_tokens': bad})


if __name__ == '__main__':
    unittest.main(verbosity=2)
