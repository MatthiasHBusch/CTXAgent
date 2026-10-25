"""Offline regressions: complete defaults, scoped pins and total deep budgets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ctxagent import config, store, defaults, blobs
from ctxagent.agent import AgentSession
from ctxagent.config import SessionConfig, CompactionConfig
from ctxagent.compaction import CompactionEngine
from ctxagent.compaction.deep_plan import plan
from ctxagent.segments import Conversation, K_SYSTEM, K_USER, K_ASSISTANT, K_TOOL_CALL, K_TOOL_RESULT, K_SUMMARY
from tests.test_compaction import pairing_ok


class FixTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module, name, value in ((store, 'DB_PATH', self.root/'test.db'),
                                    (config, 'DATA_DIR', self.root),
                                    (config, 'BLOB_DIR', self.root/'blobs'),
                                    (blobs, 'BLOB_DIR', self.root/'blobs')):
            p = patch.object(module, name, value); p.start(); self.addCleanup(p.stop)
        store.init()

    def trajectory(self, large_last=False):
        c = Conversation()
        c.new('system', K_SYSTEM, 'System instructions')
        c.new('user', K_USER, 'Implement the numerical solver')
        for i in range(36):
            c.new('assistant', K_TOOL_CALL, tool_calls=[{
                'id': f'call-{i}', 'type': 'function',
                'function': {'name': 'read_file', 'arguments': '{}'}}])
            c.new('tool', K_TOOL_RESULT, (f'line {i} data ' * (8000 if large_last and i == 35 else 400)),
                  tool_call_id=f'call-{i}', tool_name='read_file')
        return c

    def engine(self, **kw):
        cfg = CompactionConfig(document_compaction_enabled=False, deep_target_ratio=.1, **kw)
        complete = AsyncMock(return_value='## Current state\nSolver inspected. Next: add the regression test.')
        return CompactionEngine('test', cfg, complete), complete

    async def test_full_defaults_and_nested_override_and_empty_tools(self):
        cfg = SessionConfig(model='my/model', context_window=200000, max_output_tokens=8000,
            pricing_mode='flex_only', provider_policy='any', reasoning_effort='low',
            temperature=.3, enabled_tools=['read_file'], workdirs=[str(self.root)])
        cfg.compaction.document_compaction_enabled = False
        cfg.compaction.summary_pricing_mode = 'prefer_flex'
        defaults.save(cfg.to_dict())
        clone = defaults.create_config()
        self.assertEqual(clone.to_dict(), cfg.to_dict())
        other = defaults.create_config({'compaction': {'hot_zone_tokens': 12000}, 'enabled_tools': []})
        self.assertFalse(other.compaction.document_compaction_enabled)
        self.assertEqual(other.compaction.summary_pricing_mode, 'prefer_flex')
        session = AgentSession('empty', other, Conversation())
        self.assertEqual(session.tool_names(), [])
        result = await session._exec_tool('recall', {'segment_id': session.conv.segments[0].id})
        self.assertTrue(result.error)
        self.assertIn('not enabled', result.text)
        other.workdirs.clear()
        self.assertEqual(defaults.get(), cfg.to_dict())

    async def test_api_defaults_use_saved_not_active_other_chat(self):
        import httpx
        from ctxagent.server import app
        cfg = SessionConfig(model='saved/model', pricing_mode='flex_only').to_dict()
        cfg['compaction']['document_compaction_enabled'] = False
        cid = store.create_chat('source', cfg)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            r = await client.post(f'/api/chats/{cid}/defaults')
            self.assertEqual(r.status_code, 200)
            created = await client.post('/api/chats', json={'title': 'new'})
            new = store.get_chat(created.json()['id'])
            self.assertEqual(new['config'], cfg)
            disabled = await client.post('/api/chats', json={'config': {'enabled_tools': []}})
            self.assertEqual(store.get_chat(disabled.json()['id'])['config']['enabled_tools'], [])
            self.assertEqual((await client.get('/api/health')).json()['defaults'], cfg)

    async def test_new_chat_inherits_full_saved_settings_without_runtime_state(self):
        import httpx
        from ctxagent.server import app
        template = SessionConfig(model='template/model').to_dict()
        defaults.save(template)
        cfg = SessionConfig(model='source/model', context_window=200000,
            max_output_tokens=8000, max_tool_iterations=7, temperature=.3,
            reasoning_effort='low', pricing_mode='prefer_flex', provider_policy='any',
            system_prompt='Custom instructions', workdirs=[str(self.root)],
            enabled_tools=['read_file'], agent_max_parallel=3,
            agent_max_depth=1, agent_timeout_seconds=123).to_dict()
        cfg['compaction'].update(document_compaction_enabled=False,
            hot_zone_tokens=12345, summary_model='source/summary',
            summary_pricing_mode='flex_only', rewrite_interval_batches=10)
        cid = store.create_chat('source', cfg)
        store.save_chat(cid, conversation=self.trajectory().to_dict(),
            usage={'requests':2, 'cost':.123}, compaction_log=[{'tier':'test'}])
        original = store.get_chat(cid)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/chats', json={'source_chat':cid})
        self.assertEqual(response.status_code, 200)
        new = store.get_chat(response.json()['id'])
        self.assertNotEqual(new['id'], cid)
        self.assertEqual(new['config'], cfg)
        self.assertEqual(new['title'], 'New chat')
        self.assertEqual(new['conversation']['segments'], [])
        self.assertEqual(new['conversation']['frozen_upto'], 0)
        self.assertFalse(new['conversation'].get('workspace'))
        self.assertEqual(new['usage'], {})
        self.assertEqual(new['compaction_log'], [])
        new['config']['workdirs'].clear()
        new['config']['compaction']['hot_zone_tokens'] = 999
        store.save_chat(new['id'], config=new['config'])
        self.assertEqual(store.get_chat(cid), original)
        self.assertEqual(defaults.get(), template)

    async def test_inherited_settings_allow_nested_overrides_and_empty_tools(self):
        import httpx
        from ctxagent.server import app
        cfg = SessionConfig(model='source/model', enabled_tools=[],
            workdirs=[str(self.root)], pricing_mode='prefer_flex').to_dict()
        cfg['compaction']['summary_model'] = 'source/summary'
        cid = store.create_chat('source', cfg)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/chats', json={'source_chat':cid,
                'config':{'compaction':{'hot_zone_tokens':12345}}})
            self.assertEqual(response.status_code, 200)
            bad = await client.post('/api/chats', json={'source_chat':cid,
                'config':{'pricing_mode':'invalid'}})
            self.assertEqual(bad.status_code, 422)
        expected = defaults.merged(cfg, {'compaction':{'hot_zone_tokens':12345}})
        self.assertEqual(store.get_chat(response.json()['id'])['config'], expected)
        self.assertEqual(store.get_chat(cid)['config'], cfg)

    async def test_missing_inheritance_source_does_not_create_chat(self):
        import httpx
        from ctxagent.server import app
        before = store.list_chats()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/chats', json={'source_chat':'missing'})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(store.list_chats(), before)

    async def test_pending_patch_once_preserves_history_and_other_settings(self):
        cfg = SessionConfig(pricing_mode='flex_only').to_dict()
        cid = store.create_chat('source', cfg)
        store.set_setting(defaults.PENDING, {cid: {'compaction': {'document_compaction_enabled': False}}})
        before = store.get_chat(cid)
        defaults.apply_pending(); defaults.apply_pending()
        after = store.get_chat(cid)
        self.assertEqual(before['conversation'], after['conversation'])
        self.assertEqual(after['config']['pricing_mode'], 'flex_only')
        self.assertFalse(after['config']['compaction']['document_compaction_enabled'])
        self.assertIsNone(store.get_setting(defaults.PENDING))

    async def test_document_master_switch_blocks_all_three_passes_under_pressure(self):
        c = self.trajectory()
        e, fn = self.engine(deep_enabled=False, rewrite_interval_batches=1)
        before = c.to_messages()
        with patch('ctxagent.compaction.engine.dedup.run') as d, \
             patch('ctxagent.compaction.engine.masking.run') as m, \
             patch('ctxagent.compaction.engine.summarize.condense_batch') as s:
            for _ in range(8):
                await e.run(c, 20000)
            d.assert_not_called(); m.assert_not_called(); s.assert_not_called()
        self.assertEqual(before, c.to_messages())
        self.assertGreater(c.frozen_upto, 0)
        fn.assert_not_called()

    async def test_deep_targets_remaining_total_not_tail_alone(self):
        c = self.trajectory(); e, fn = self.engine(summary_max_tokens=3000)
        before = c.total_tokens() + 4000
        r = await e._deep(c, 179000, .2, 4000, '', True)
        self.assertTrue(r['target_met'])
        self.assertLessEqual(c.total_tokens()+4000, int(before*.1))
        self.assertTrue(pairing_ok(c))
        self.assertTrue(c.archive)
        fn.assert_awaited_once()
        self.assertIn('STRICT OUTPUT BUDGET', fn.call_args.kwargs['messages'][0]['content'])

    async def test_recall_protection_archived_but_explicit_pin_retained(self):
        c = self.trajectory(); e, fn = self.engine()
        original = c.segments[3]; original.restore()
        self.assertEqual(original.pin_scope, 'recall')
        pinned = c.segments[5]; pinned.pinned = True; pinned.pin_scope = 'manual'
        r = await e._deep(c, 179000, .9, 100, '', True)
        self.assertIsNone(c.by_id(original.id))
        self.assertEqual(c.archived_by_id(original.id).body, original.body)
        self.assertIs(c.by_id(pinned.id), pinned)
        self.assertTrue(pairing_ok(c))
        self.assertGreaterEqual(r['pinned_retained'], 2)

    async def test_legacy_recall_evidence_in_archive_migrates_but_unknown_pin_does_not(self):
        c = self.trajectory()
        recalled = c.segments[3]; recalled.pinned = True; recalled.pin_scope = 'legacy'
        unknown = c.segments[5]; unknown.pinned = True; unknown.pin_scope = 'legacy'
        evidence = Conversation().new('assistant', K_TOOL_CALL, tool_calls=[{'function': {
            'name': 'recall', 'arguments': json.dumps({'segment_id': recalled.id})}}])
        c.archive_segments([evidence], 'test')
        c.migrate_legacy_recall_pins()
        self.assertEqual(recalled.pin_scope, 'recall')
        self.assertEqual(unknown.pin_scope, 'legacy')
        clone = Conversation.from_dict(c.to_dict())
        self.assertEqual(clone.by_id(recalled.id).pin_scope, 'recall')

    async def test_huge_latest_group_reports_floor_and_keeps_pairing(self):
        c = self.trajectory(large_last=True); e, fn = self.engine()
        last = c.segments[-1]
        r = await e._deep(c, 179000, 1.0, 4500, '', True)
        self.assertFalse(r['target_met'])
        self.assertIn('newest', r['warning'])
        self.assertIs(c.by_id(last.id), last)
        self.assertTrue(pairing_ok(c))

    async def test_oversized_summary_bounded_without_paid_retry(self):
        c = self.trajectory(); e, fn = self.engine(summary_max_tokens=3000)
        fn.return_value = 'Too verbose summary of every event.\n' * 10000
        before = c.total_tokens()
        r = await e._deep(c, 179000, .9, 0, '', True)
        self.assertTrue(r['summary_clipped'])
        self.assertLessEqual(c.total_tokens(), int(before*.1))
        fn.assert_awaited_once()

    async def test_failed_summary_preserves_original_context(self):
        c = self.trajectory(); e, fn = self.engine()
        fn.side_effect = RuntimeError('capacity')
        before = c.to_messages()
        r = await e._deep(c, 179000, .9, 0, '', True)
        self.assertIn('error', r)
        self.assertEqual(before, c.to_messages())
        self.assertFalse(c.archive)

    async def test_recent_users_not_permanently_pinned(self):
        c = Conversation(); c.new('system', K_SYSTEM, 'system')
        e, fn = self.engine(deep_enabled=False)
        for _ in range(8):
            c.new('user', K_USER, 'a new question')
            await e.run(c, 179000)
        self.assertFalse(any(s.pinned for s in c if s.kind == K_USER))

    def test_target_ratio_validated(self):
        for bad in (float('nan'), float('inf'), 0, 10, -.1):
            with self.assertRaises(ValueError):
                CompactionConfig.from_dict({'deep_target_ratio': bad})


if __name__ == '__main__':
    unittest.main(verbosity=2)
