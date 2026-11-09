"""A new chat's settings (2026-11-09): Opus 5.5 at xhigh, 700k window, self-steered with deep
compaction at 90 %, unlimited output, every tool on; document_call is not a user tool; the
model id follows the key the user has; the chat's System instructions follow the settings."""
import asyncio
import os
import unittest
from unittest.mock import patch

from ctxagent import config, defaults, llm, tools
from ctxagent.config import CompactionConfig, SessionConfig
from tests.test_settings import Isolated


class Defaults(unittest.TestCase):
    def test_new_chat_settings(self):
        for cfg in (SessionConfig(), SessionConfig.from_dict({})):
            self.assertEqual((cfg.model, cfg.reasoning_effort, cfg.context_window, cfg.max_output_tokens),
                             ('anthropic/claude-opus-5.5', 'xhigh', 700_000, 0))
            self.assertTrue(cfg.compaction.self_steered)
            self.assertEqual(cfg.compaction.deep_trigger_ratio, 0.9)
            self.assertEqual(set(cfg.enabled_tools), {t['name'] for t in tools.catalogue()})
            self.assertIn('compact_context', cfg.active_tools())
        # The engine's own defaults are unchanged; a stored chat keeps what it stored.
        self.assertEqual(CompactionConfig().rewrite_policy, 'auto')
        old = SessionConfig.from_dict({'model': 'x/y', 'max_output_tokens': 32_000, 'compaction': {}})
        self.assertEqual((old.model, old.max_output_tokens, old.compaction.rewrite_policy), ('x/y', 32_000, 'auto'))

    def test_document_call_is_not_a_user_tool(self):
        self.assertNotIn('document_call', [t['name'] for t in tools.catalogue()])
        self.assertNotIn('document_call', tools.DEFAULT_TOOLS)

    def test_unlimited_output(self):
        cfg = SessionConfig()
        self.assertEqual(cfg.output_reserve(), config.UNLIMITED_OUTPUT_RESERVE)
        self.assertEqual(cfg.effective_window(), 700_000 - 32_000 - 13_000)
        self.assertEqual(cfg.request_max_tokens(100_000), 128_000)          # the model's maximum
        self.assertEqual(cfg.request_max_tokens(690_000), 8_000)            # what the window has room for
        self.assertEqual(cfg.request_max_tokens(699_000), 4_096)
        cfg.model = 'openai/gpt-6.1-sol'
        self.assertIsNone(cfg.request_max_tokens(100_000))                  # OpenRouter decides
        cfg.max_output_tokens = 16_000
        self.assertEqual(cfg.request_max_tokens(100_000), 16_000)
        self.assertEqual(cfg.effective_window(), 700_000 - 16_000 - 13_000)
        self.assertEqual(config.model_output_limit('claude-haiku-4-5'), 64_000)

    def test_openrouter_request_without_max_tokens(self):
        from unittest.mock import AsyncMock
        sent = []

        def route(payload, plan):
            sent.append(payload)
            raise ConnectionAbortedError('captured')    # nothing is sent

        async def run():
            try:
                async for _ in llm.stream_chat(model='openai/gpt-6.1-sol',
                                               messages=[{'role': 'user', 'content': 'hi'}], max_tokens=None):
                    pass
            except ConnectionAbortedError:
                pass
        with patch('ctxagent.llm._apply_route', route), patch('ctxagent.llm._headers', return_value={}),                 patch('ctxagent.llm.flex.plans', AsyncMock(return_value=[{'tier': 'standard'}])):
            asyncio.run(run())
        self.assertTrue(sent)
        self.assertNotIn('max_tokens', sent[0])

    def test_provider_counterpart(self):
        self.assertEqual(config.provider_counterpart('anthropic/claude-opus-5.5'), 'claude-opus-5-5')
        self.assertEqual(config.provider_counterpart('claude-opus-5-5'), 'anthropic/claude-opus-5.5')
        self.assertEqual(config.provider_counterpart('claude-sonnet-5-5-20260901'), 'anthropic/claude-sonnet-5.5')
        self.assertIsNone(config.provider_counterpart('openai/gpt-6.1-sol'))


class NewChats(Isolated):
    def test_model_follows_the_key_the_user_has(self):
        os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-api03-test-000000000000000000'
        self.addCleanup(os.environ.pop, 'ANTHROPIC_API_KEY', None)
        self.assertEqual(defaults.create_config().model, 'claude-opus-5-5')
        os.environ['OPENROUTER_API_KEY'] = 'sk-or-v1-' + '0' * 64
        self.addCleanup(os.environ.pop, 'OPENROUTER_API_KEY', None)
        self.assertEqual(defaults.create_config().model, 'anthropic/claude-opus-5.5')   # OpenRouter key: as set

    def test_system_instructions_follow_self_steering(self):
        c = self.client()
        cid = c.post('/api/chats', json={}).json()['id']
        on = c.patch(f'/api/chats/{cid}/config', json={'config': {'compaction': {'rewrite_policy': 'agent'}}}).json()
        off = c.patch(f'/api/chats/{cid}/config', json={'config': {'compaction': {'rewrite_policy': 'auto'}}}).json()
        self.assertNotEqual(on['system'], off['system'])
        self.assertIn('compact_context', on['system'])
        self.assertNotIn('compact_context', off['system'])
        segs = c.get(f'/api/chats/{cid}').json()['segments']
        self.assertEqual(segs[0]['body'], off['system'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
