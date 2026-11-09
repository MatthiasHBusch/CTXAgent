"""Review-before-freeze and bounded-delivery regressions, without model calls."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from ctxagent import blobs, config, result_delivery
from ctxagent.compaction import CompactionEngine, curate, freeze_review
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_ASSISTANT, K_MEDIA
from tests.test_compaction import fake_complete, pairing_ok
from tests.test_productivity import make_session


def settings(**kw):
    d = dict(review_before_freeze=True, hot_zone_tokens=1000,
             protected_recent_observations=2, protected_recent_batches=0, rewrite_interval_batches=5,
             rewrite_max_new_tokens=1000000, deep_enabled=False)
    d.update(kw)
    return CompactionConfig(**d)


def batch(conv, n=1, body=None):
    ids = [f'call-{len(conv)}-{i}' for i in range(n)]
    call = conv.new('assistant', K_TOOL_CALL, '', tool_calls=[
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
        for cid in ids])
    results = [conv.new('tool', K_TOOL_RESULT, body or f'unique {cid}\n' + 'result evidence ' * 1400,
                        tool_name='read_file', tool_call_id=cid) for cid in ids]
    return call, results


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_mode_freezes_reviewed_history_and_never_rewrites(self):
        c = Conversation()
        e = CompactionEngine('engine-review', settings(), fake_complete)
        for i in range(5):
            _, res = batch(c)
            before = c.to_messages()
            await e.run(c, 1000000)
            self.assertEqual(c.to_messages(), before)
            self.assertFalse(res[0].frozen)                 # the unseen batch stays out
        self.assertGreater(c.frozen_upto, 0)
        protected = [s for s in c.segments if s.kind == K_TOOL_RESULT][-2:]
        self.assertTrue(all(not s.frozen for s in protected))
        self.assertEqual(c.compaction_state['reviewed_through_seq'], c.segments[c.frozen_upto - 1].seq)
        self.assertTrue(pairing_ok(c))

    async def test_self_steered_prefix_waits_for_curate(self):
        c = Conversation()
        cc = settings(rewrite_policy='agent')
        e = CompactionEngine('agent-review', cc, fake_complete)
        for _ in range(6):
            batch(c)
            await e.run(c, 1000000)
        self.assertEqual(c.frozen_upto, 0)
        rep = curate.curate(c, cc, 't', drop=[])
        self.assertGreater(rep['frozen_segments'], 0)
        self.assertGreater(c.frozen_upto, 0)
        self.assertTrue(pairing_ok(c))

    async def test_context_api_uses_actual_pending_review_boundary(self):
        from ctxagent import server
        c = Conversation()
        for _ in range(7):
            batch(c)
        session = make_session(c)
        session.cfg.compaction = session.engine.cfg = settings()
        self.assertGreater(c.compute_hot_start(1000), 0)
        with patch('ctxagent.server.agent.get_session', return_value=session):
            report = await server.context('test')
        self.assertEqual(report['hot_start'], 0)
        self.assertTrue(all(s['hot'] for s in report['segments']))

    async def test_frozen_bytes_never_change(self):
        c = Conversation()
        for _ in range(8):
            batch(c)
        e = CompactionEngine('order-review', settings(), fake_complete)
        await e.run(c, 1000000)
        frozen = copy.deepcopy(c.to_messages()[:c.frozen_upto])
        boundary = c.frozen_upto
        for _ in range(3):
            batch(c)
            await e.run(c, 1000000)
        self.assertEqual(frozen, c.to_messages()[:boundary])
        self.assertGreater(c.frozen_upto, boundary)
        self.assertTrue(pairing_ok(c))

    async def test_whole_unseen_large_parallel_batch_survives_pressure(self):
        c = Conversation()
        for _ in range(5):
            batch(c)
        call, fresh = batch(c, 12)
        before = [s.to_message() for s in [call, *fresh]]
        e = CompactionEngine('fresh-review', settings(), fake_complete)
        r = await e.run(c, 1000)
        self.assertGreater(r['pressure'], 1)
        self.assertEqual(before, [s.to_message() for s in [call, *fresh]])
        self.assertTrue(all(not s.frozen for s in [call, *fresh]))
        self.assertTrue(pairing_ok(c))

    async def test_protected_recent_not_frozen(self):
        c = Conversation()
        for _ in range(6):
            batch(c, body='identical observations ' * 1000)
        protected = c.segments[-4:]
        before = [s.to_message() for s in protected]
        await CompactionEngine('protect-review', settings(), fake_complete).run(c, 1000000)
        self.assertEqual(before, [s.to_message() for s in protected])
        self.assertTrue(all(not s.frozen for s in protected))

    async def test_recalled_and_manual_pins_stay_verbatim(self):
        c = Conversation()
        _, old = batch(c)
        old[0].pinned, old[0].pin_scope = True, 'recall'
        _, manual = batch(c)
        manual[0].pinned = True
        for _ in range(6):
            batch(c)
        before = [s.to_message() for s in [old[0], manual[0]]]
        await CompactionEngine('pins-review', settings(), fake_complete).run(c, 1000000)
        self.assertEqual(before, [s.to_message() for s in [old[0], manual[0]]])

    async def test_visuals_stay_native_and_may_freeze(self):
        c = Conversation()
        media = []
        for i in range(6):
            batch(c)
            media.append(c.new('user', K_MEDIA, 'visual evidence', images=[{'id': f'image-{i}'}]))
        before = [s.to_message() for s in media]
        await CompactionEngine('media-review', settings(), fake_complete).run(c, 1000000)
        self.assertEqual(before, [s.to_message() for s in media])
        self.assertTrue(all(s.state == 'original' for s in media))
        self.assertTrue(media[0].frozen)
        self.assertFalse(media[-1].frozen)

    async def test_old_retained_images_do_not_pin_the_entire_mutable_suffix(self):
        c = Conversation()
        media = [c.new('user', K_MEDIA, 'retained image', images=[{'id':f'old-{i}'}]) for i in range(2)]
        original = [copy.deepcopy(s.to_message()) for s in media]
        for _ in range(12):
            batch(c)
        e = CompactionEngine('old-media-cut', settings(), fake_complete)
        await e.run(c, 1000000)
        self.assertGreater(c.frozen_upto, 2)
        self.assertTrue(all(s.frozen and s.state == 'original' for s in media))
        self.assertEqual([s.to_message() for s in media], original)
        self.assertFalse(c.segments[-1].frozen)  # unseen current result still protected

    async def test_legacy_frozen_prefix_not_thawed_by_migration(self):
        c = Conversation()
        batch(c)
        c.freeze_prefix(2)
        original = c.segments[1].body
        for _ in range(6):
            batch(c)
        await CompactionEngine('migrate-review', settings(), fake_complete).run(c, 1000000)
        self.assertEqual(c.segments[1].body, original)
        self.assertTrue(c.segments[1].frozen)

    async def test_state_and_boundary_survive_reload(self):
        c = Conversation()
        e = CompactionEngine('reload-review', settings(), fake_complete)
        for _ in range(4):
            batch(c)
            await e.run(c, 1000000)
        boundary = c.frozen_upto
        c = Conversation.from_dict(c.to_dict())
        self.assertEqual(c.frozen_upto, boundary)
        batch(c)
        await CompactionEngine('reload-review', settings(), fake_complete).run(c, 1000000)
        self.assertGreater(c.frozen_upto, boundary)
        self.assertEqual(c.compaction_state['reviewed_through_seq'], c.segments[c.frozen_upto-1].seq)
        clone = Conversation.from_dict(c.to_dict())
        self.assertEqual(clone.frozen_upto, c.frozen_upto)

    async def test_deep_pressure_retains_unseen_group(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'system')
        for _ in range(10):
            batch(c)
        call, fresh = batch(c, 8)
        before = [s.to_message() for s in [call, *fresh]]
        cc = settings(deep_enabled=True, deep_cooldown_turns=0, deep_min_gain_tokens=0)
        r = await CompactionEngine('deep-review', cc, fake_complete).run(c, 1000)
        self.assertTrue(r['deep'])
        self.assertEqual(before, [s.to_message() for s in [call, *fresh]])
        self.assertTrue(pairing_ok(c))

    def test_cut_snaps_back_to_whole_group(self):
        c = Conversation()
        batch(c, 5)
        c.new('assistant', K_ASSISTANT, 'consumed')
        cc = settings(hot_zone_tokens=5000, protected_recent_observations=0)
        self.assertTrue(0 < c.compute_hot_start(cc.hot_zone_tokens) < 6)
        self.assertEqual(freeze_review.plan(c, cc)['end'], 0)


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
        c.freeze_prefix(len(c))
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
        self.assertTrue(cc.review_before_freeze)
        self.assertEqual(cc.initial_result_max_tokens, 4000)
        for bad in (-1, 100, 100001, float('nan')):
            with self.assertRaises((ValueError, OverflowError)):
                CompactionConfig.from_dict({'initial_result_max_tokens': bad})
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'review_before_freeze': 'true'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
