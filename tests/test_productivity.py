"""Offline integration/regression tests. Run via tests.run_all for isolated data."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ctxagent import artifacts, config, jobs, llm, tools
from ctxagent.agent import AgentSession
from ctxagent.compaction import CompactionEngine, dedup
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.observations import covers
from ctxagent.segments import (Conversation, Segment, K_SYSTEM, K_USER, K_TOOL_CALL,
                               K_TOOL_RESULT, K_MEDIA, S_SUPERSEDED)
from tests.test_compaction import build_conv, fake_complete, pairing_ok


def add_result(conv, result):
    return conv.new('tool', K_TOOL_RESULT, result.text, resource_key=result.resource_key,
                    tool_name='read_file', detail=result.detail, channel=result.channel,
                    observation=result.observation)


def make_session(conv=None, root=None):
    cfg = SessionConfig(compaction=CompactionConfig(rewrite_interval_batches=1, enabled=False),
                        workdirs=[str(root)] if root else [])
    session = AgentSession('test-session', cfg, conv or Conversation())
    session.persist = lambda: None
    return session


class IntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sb = tools.Sandbox([str(self.root)])
        self.file = self.root / 'sample.txt'
        self.file.write_text('\n'.join(f'{i}: scientific result and context ' * 8 for i in range(250)), encoding='utf-8')

    async def test_disjoint_ranges_preserved_containing_range_replaces(self):
        c = Conversation()
        old = add_result(c, await tools.t_read_file(self.sb, 'sample.txt', 1, 50))
        add_result(c, await tools.t_read_file(self.sb, 'sample.txt', 100, 50))
        dedup.run(c, 0)
        self.assertEqual(old.state, 'original')
        add_result(c, await tools.t_read_file(self.sb, 'sample.txt', 1, 200))
        dedup.run(c, 0)
        self.assertEqual(old.state, S_SUPERSEDED)

    async def test_disjoint_document_ranges_and_representation(self):
        c = Conversation()
        first = add_result(c, await tools.t_read_document(self.sb, 'sample.txt', 1, 40))
        add_result(c, await tools.t_read_document(self.sb, 'sample.txt', 100, 40))
        add_result(c, await tools.t_read_file(self.sb, 'sample.txt', 1, 250))
        dedup.run(c, 0)
        self.assertEqual(first.state, 'original')

    async def test_external_edit_preserved(self):
        c = Conversation()
        first = add_result(c, await tools.t_read_file(self.sb, 'sample.txt'))
        self.file.write_text('changed\n' + self.file.read_text(encoding='utf-8'), encoding='utf-8')
        add_result(c, await tools.t_read_file(self.sb, 'sample.txt'))
        dedup.run(c, 0)
        self.assertEqual(first.state, 'original')

    async def test_focus_different_selectors_preserved(self):
        p = self.root / 'module.py'
        p.write_text('def alpha():\n' + '    x = 12345\n' * 80 + '\ndef beta():\n' + '    y = 67890\n' * 80)
        c = Conversation()
        first = add_result(c, await tools.t_focus_file(self.sb, 'module.py', 'alpha'))
        add_result(c, await tools.t_focus_file(self.sb, 'module.py', 'beta'))
        dedup.run(c, 0)
        self.assertEqual(first.state, 'original')
        add_result(c, await tools.t_focus_file(self.sb, 'module.py', ['alpha', 'beta']))
        dedup.run(c, 0)
        self.assertEqual(first.state, S_SUPERSEDED)

    async def test_unknown_and_incomplete_coverage(self):
        self.assertFalse(covers({}, {}))
        obs = {'version': 'v', 'representation': 'x', 'complete': True, 'coverage': {'kind': 'whole'}}
        self.assertFalse(covers({**obs, 'complete': False}, obs))
        self.assertFalse(covers({**obs, 'version': 'other'}, obs))
        self.assertTrue(covers(obs, obs))

    async def test_deep_archive_roundtrip_and_recall(self):
        c = build_conv(n_reads=9, big_lines=100)
        old = next(s for s in c if s.kind == K_TOOL_RESULT)
        original = old.body
        engine = CompactionEngine('archive-test', CompactionConfig(rewrite_interval_batches=1, ), fake_complete)
        await engine._deep(c, 20000, 1.0, 0, '', True)
        self.assertIsNone(c.by_id(old.id))
        c = Conversation.from_dict(json.loads(c.dumps()))
        self.assertEqual(c.archived_by_id(old.id).body, original)
        result = await make_session(c)._exec_tool('recall', {'segment_id': old.id})
        self.assertIn(original, result.text)
        self.assertTrue(result.pinned)
        self.assertTrue(pairing_ok(c))

    async def test_deep_pins_keep_tool_group(self):
        c = build_conv(n_reads=9, big_lines=100)
        old = next(s for s in c if s.kind == K_TOOL_RESULT)
        old.pinned = True
        before = old.body
        engine = CompactionEngine('pin-test', CompactionConfig(rewrite_interval_batches=1, ), fake_complete)
        report = await engine._deep(c, 20000, 1.0, 0, '', True)
        self.assertEqual(c.by_id(old.id).body, before)
        self.assertGreaterEqual(report['pinned_retained'], 2)
        self.assertTrue(pairing_ok(c))

    async def test_deep_final_batch_never_orphaned(self):
        # Previously a cut snapped to EOF was clamped inside the final result batch.
        c = Conversation()
        c.new('system', K_SYSTEM, 'system')
        for i in range(5):
            c.new('assistant', 'assistant_text', 'old context ' * 500)
        c.new('assistant', K_TOOL_CALL, tool_calls=tool_calls())
        c.new('tool', K_TOOL_RESULT, 'first result ' * 500, tool_call_id='first')
        c.new('tool', K_TOOL_RESULT, 'last result ' * 500, tool_call_id='second')
        engine = CompactionEngine('cut-test', CompactionConfig(rewrite_interval_batches=1, ), fake_complete)
        await engine._deep(c, 20000, 1.0, 0, '', True)
        self.assertTrue(pairing_ok(c))
        self.assertEqual([s.tool_call_id for s in c if s.kind == K_TOOL_RESULT], ['first', 'second'])

    async def test_all_pinned_no_deletion(self):
        c = build_conv(n_reads=5, big_lines=100)
        for seg in c:
            seg.pinned = True
        ids = [s.id for s in c]
        engine = CompactionEngine('all-pin-test', CompactionConfig(rewrite_interval_batches=1, ), fake_complete)
        await engine._deep(c, 20000, 1.0, 0, '', True)
        self.assertEqual(ids, [s.id for s in c])

    async def test_cancellation_closes_batch_and_skips_second_tool(self):
        session = make_session()
        calls = []
        async def execute(name, args):
            calls.append(name)
            session.cancel.set()
            return tools.ToolResult('completed')
        session._exec_tool = execute
        with patch('ctxagent.agent.llm.stream_chat', fake_stream):
            async for _ in session.run_turn('cancel test'):
                pass
        self.assertEqual(len(calls), 1)
        self.assertTrue(pairing_ok(session.conv))
        self.assertIn('not executed', [s for s in session.conv if s.kind == K_TOOL_RESULT][-1].body)

    async def test_disconnect_repairs_batch_and_unlocks(self):
        session = make_session()
        with patch('ctxagent.agent.llm.stream_chat', fake_stream):
            stream = session.run_turn('disconnect test')
            async for event in stream:
                if event['type'] == 'segment' and event['segment']['kind'] == K_TOOL_CALL:
                    break
            await stream.aclose()
        self.assertTrue(pairing_ok(session.conv))
        self.assertFalse(session.turn_lock.locked())
        self.assertEqual(session.conv.repair_tool_pairs(), 0)

    async def test_unlimited_default_exceeds_forty_iterations(self):
        session = make_session()
        self.assertEqual(session.cfg.max_tool_iterations, 0)
        requests = 0
        async def stream(**kw):
            nonlocal requests
            requests += 1
            if requests <= 45:
                yield {'type': 'tool_calls', 'tool_calls': [{
                    'id': f'call-{requests}', 'type': 'function',
                    'function': {'name': 'read_file', 'arguments': '{}'}}]}
            else:
                yield {'type': 'text', 'text': 'finished after 45 tools'}
            yield {'type': 'done', 'finish_reason': 'stop'}
        async def execute(*args): return tools.ToolResult('result')
        session._exec_tool = execute
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [event async for event in session.run_turn('long task')]
        self.assertEqual(requests, 46)
        self.assertFalse(any(e['type'] == 'error' for e in events))
        self.assertTrue(pairing_ok(session.conv))

    async def test_optional_iteration_limit_is_serialized_and_enforced(self):
        cfg = SessionConfig.from_dict({'max_tool_iterations': 2})
        self.assertEqual(SessionConfig.from_dict(cfg.to_dict()).max_tool_iterations, 2)
        session = make_session()
        session.cfg.max_tool_iterations = 2
        async def execute(*args): return tools.ToolResult('result')
        session._exec_tool = execute
        with patch('ctxagent.agent.llm.stream_chat', fake_stream):
            events = [event async for event in session.run_turn('limited')]
        self.assertEqual(len([e for e in events if e['type'] == 'tool_result']), 4)
        self.assertIn('configured limit of 2', next(e['error'] for e in events if e['type'] == 'error'))
        self.assertFalse(session.turn_lock.locked())

    async def test_hot_zone_reports_actual_boundary_before_and_after_freeze(self):
        c = build_conv(n_reads=12, big_lines=150)
        cfg = CompactionConfig(rewrite_interval_batches=1, hot_zone_tokens=10000)
        engine = CompactionEngine('budget-test', cfg, fake_complete)
        before = engine.stats(c, 200000)
        self.assertEqual(before['hot_zone_budget'], 10000)
        # Inspection must not claim a prospective boundary is already frozen.
        self.assertEqual(before['frozen_tokens'], 0)
        self.assertEqual(before['hot_tokens'], c.total_tokens() - 3)
        await engine.run(c, 200000)
        after = engine.stats(c, 200000)
        self.assertLessEqual(after['hot_tokens'], 10000)
        self.assertEqual(CompactionConfig.from_dict(cfg.to_dict()).hot_zone_tokens, 10000)

    async def test_concurrent_turn_rejected(self):
        session = make_session()
        stream = session.run_turn('first')
        await anext(stream)
        other = session.run_turn('second')
        with self.assertRaisesRegex(RuntimeError, 'already running'):
            await anext(other)
        await stream.aclose()
        self.assertFalse(session.turn_lock.locked())

    async def test_reload_repairs_missing_results(self):
        c = Conversation()
        c.new('assistant', K_TOOL_CALL, tool_calls=tool_calls())
        c.new('user', K_USER, 'next request')
        s = make_session(Conversation.from_dict(c.to_dict()))
        self.assertTrue(pairing_ok(s.conv))
        messages = s.conv.to_messages()
        self.assertEqual([m['role'] for m in messages], ['system', 'assistant', 'tool', 'tool', 'user'])

    async def test_archive_blob_exact_text_and_id_validation(self):
        from ctxagent.blobs import put_blob, get_blob
        original = 'alpha\r\nbeta\n\u03bb'
        handle = put_blob('blob-test', 'segment', original)
        self.assertEqual(get_blob(handle), original)
        self.assertNotEqual(put_blob('blob-test', 'segment', 'different'), handle)
        with self.assertRaises(ValueError):
            get_blob('../escape')

    async def test_resolve_traversal_and_link(self):
        with self.assertRaises(tools.ToolError):
            self.sb.resolve('../outside.txt')
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'secret.txt'
            target.write_text('secret')
            link = self.root / 'link.txt'
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest('Host does not permit file symlink creation')
            with self.assertRaises(tools.ToolError):
                await tools.t_read_file(self.sb, 'link.txt')


def tool_calls():
    return [{'id': name, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
            for name in ('first', 'second')]


async def fake_stream(**kw):
    yield {'type': 'tool_calls', 'tool_calls': tool_calls()}
    yield {'type': 'done', 'finish_reason': 'tool_calls'}


class VisualTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'plot.png'
        Image.new('RGB', (800, 600), 'red').save(self.path)

    async def test_image_native_payload_and_persistence(self):
        images = artifacts.view(self.path, 512)
        self.assertEqual((images[0]['width'], images[0]['height']), (512, 384))
        c = Conversation()
        media = c.new('user', K_MEDIA, 'plot evidence', images=images)
        self.assertLess(len(c.dumps()), 2000)
        self.assertGreater(media.tokens(), 2000)
        restored = Conversation.from_dict(c.to_dict())
        messages = restored.to_messages()
        expanded = artifacts.expand_messages(messages)
        self.assertTrue(expanded[0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(messages[0]['content'][1]['type'], 'image_ref')
        self.assertEqual(artifacts.view(self.path, 512)[0]['id'], images[0]['id'])
        with self.assertRaises(ValueError):
            artifacts.artifact_path('../secret')

    async def test_pdf_page_order_bounds_and_pixels(self):
        import fitz
        pdf = self.root / 'paper.pdf'
        with fitz.open() as doc:
            for title in ['first', 'second']:
                doc.new_page().insert_text((72, 72), title)
            doc.save(pdf)
        images = artifacts.render(pdf, [2, 1], 72)
        self.assertEqual([i['page'] for i in images], [2, 1])
        self.assertNotEqual(images[0]['id'], images[1]['id'])
        for pages in ([0], [3], [], [1] * 7):
            with self.assertRaises(ValueError):
                artifacts.render(pdf, pages)
        with self.assertRaises(ValueError):
            artifacts.render(pdf, [1], 1000)
        big = self.root / 'big.pdf'
        with fitz.open() as doc:
            doc.new_page(width=10000, height=10000)
            doc.save(big)
        with self.assertRaisesRegex(ValueError, 'too large'):
            artifacts.render(big)

    async def test_media_compaction_and_recall(self):
        images = artifacts.view(self.path)
        c = Conversation()
        for i in range(3):
            c.new('user', K_MEDIA, f'image {i}', images=images)
        first = c.segments[0]
        before = first.tokens()
        engine = CompactionEngine('media-test', CompactionConfig(rewrite_interval_batches=1, deep_enabled=False,
                                                                 rewrite_cost_gate=False), fake_complete)
        await engine.run(c, 1000000)
        self.assertLess(first.tokens(), before)
        self.assertEqual(len(first.to_message()['content']), 1)
        result = await make_session(c)._exec_tool('recall', {'segment_id': first.id})
        self.assertEqual(result.images, images)
        self.assertTrue(result.pinned)

    async def test_latest_visual_batch_survives_deep_compaction(self):
        images = artifacts.view(self.path)
        c = build_conv(n_reads=5, big_lines=100)
        c.new('assistant', K_TOOL_CALL, tool_calls=tool_calls())
        c.new('tool', K_TOOL_RESULT, 'first', tool_call_id='first')
        c.new('tool', K_TOOL_RESULT, 'second', tool_call_id='second')
        latest = [c.new('user', K_MEDIA, f'page {i}', images=images) for i in range(6)]
        engine = CompactionEngine('fresh-media', CompactionConfig(rewrite_interval_batches=1, ), fake_complete)
        await engine._deep(c, 20000, 1.0, 0, '', True)
        self.assertTrue(all(c.by_id(s.id) is not None for s in latest))
        self.assertTrue(pairing_ok(c))

    async def test_media_follows_complete_batch_and_reaches_model(self):
        session = make_session(root=self.root)
        seen = []
        images = artifacts.view(self.path)
        async def stream(**kw):
            seen.append(kw['messages'])
            if len(seen) == 1:
                yield {'type': 'tool_calls', 'tool_calls': tool_calls()}
            else:
                yield {'type': 'text', 'text': 'inspected'}
            yield {'type': 'done', 'finish_reason': 'stop'}
        async def execute(*args):
            return tools.ToolResult('image ready', images=images)
        session._exec_tool = execute
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [e async for e in session.run_turn('inspect')]
        self.assertEqual(len([e for e in events if e['type'] == 'media']), 2)
        self.assertTrue(pairing_ok(session.conv))
        roles = [m['role'] for m in seen[1]]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'tool', 'user', 'user'])
        self.assertIn('image_url', json.dumps(artifacts.expand_messages(seen[1])))

    async def test_stream_chat_serializes_images_at_http_boundary(self):
        import httpx
        captured = []
        image = artifacts.view(self.path)
        c = Conversation()
        c.new('user', K_MEDIA, 'plot', images=image)
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"red"}}]}\n\ndata: [DONE]\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async def allowed(*a): return ['test-provider']
        with patch.object(llm.httpx, 'AsyncClient', return_value=client), \
             patch.object(llm, '_headers', return_value={}), \
             patch.object(llm.provmod, 'allowed_slugs', allowed):
            output = [e async for e in llm.stream_chat(model='test/vision', messages=c.to_messages())]
        self.assertEqual(output[0]['text'], 'red')
        self.assertEqual(captured[0]['messages'][0]['content'][1]['type'], 'image_url')
        self.assertNotIn('image_ref', json.dumps(captured))

    async def test_corrupt_artifact_fails_explicitly(self):
        image = artifacts.view(self.path)[0]
        p = artifacts.artifact_path(image['id'])
        original = p.read_bytes()
        try:
            p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                artifacts.expand_messages([{'role': 'user', 'content': [{'type': 'image_ref', 'image': image}]}])
        finally:
            p.write_bytes(original)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = []

    def tearDown(self):
        for identifier in self.ids:
            jobs.cancel(identifier, 'job-tests')
            self.wait(identifier)
        self.temp.cleanup()

    def start_python(self, source, timeout=30):
        script = self.root / f'run{len(self.ids)}.py'
        script.write_text(source, encoding='utf-8')
        if os.name == 'nt':
            command = '& "' + sys.executable + '" "' + str(script) + '"; exit $LASTEXITCODE'
        else:
            import shlex
            command = shlex.join([sys.executable, str(script)])
        data = jobs.start(command, self.root, 'job-tests', timeout)
        self.ids.append(data['id'])
        return data['id']

    def wait(self, identifier, max_wait=25):
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            data = jobs.status(identifier, 'job-tests')
            if data['state'] in jobs.TERMINAL:
                return data
            time.sleep(0.1)
        self.fail(f'Job did not finish: {data}')

    def test_success_and_cursor_logs(self):
        identifier = self.start_python("import sys\nprint('hello')\nprint('warning', file=sys.stderr)\n")
        data = self.wait(identifier)
        self.assertEqual(data['state'], 'succeeded', data)
        first = jobs.read_log(identifier, 'job-tests', max_bytes=3)
        second = jobs.read_log(identifier, 'job-tests', cursor=first['next_cursor'])
        self.assertEqual(first['text'] + second['text'], 'hello\r\n' if os.name == 'nt' else 'hello\n')
        self.assertIn('warning', jobs.read_log(identifier, 'job-tests', 'stderr')['text'])
        self.assertTrue(second['at_end'])
        with self.assertRaises(PermissionError):
            jobs.status(identifier, 'another-chat')
        with self.assertRaises(ValueError):
            jobs.read_log(identifier, 'job-tests', cursor=100000)
        self.assertEqual(jobs.cancel(identifier, 'job-tests')['state'], 'succeeded')

    def test_nonzero_exit(self):
        identifier = self.start_python('raise SystemExit(7)')
        data = self.wait(identifier)
        self.assertEqual(data['state'], 'failed', data)
        self.assertEqual(data['exit_code'], 7)

    def test_timeout(self):
        identifier = self.start_python('import time\ntime.sleep(60)', timeout=1)
        self.assertEqual(self.wait(identifier)['state'], 'timed_out')

    def test_streaming_cancel_and_fresh_process_lookup(self):
        identifier = self.start_python("import time\nprint('ready', flush=True)\ntime.sleep(60)")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if 'ready' in jobs.read_log(identifier, 'job-tests')['text']:
                break
            time.sleep(0.1)
        else:
            self.fail('Live output did not arrive: ' + json.dumps(jobs.status(identifier, 'job-tests')) + '\nSTDERR: ' + jobs.read_log(identifier, 'job-tests', 'stderr')['text'])
        self.assertEqual(jobs.status(identifier, 'job-tests')['state'], 'running')
        # A fresh client process has no in-memory handles, just the persisted record.
        code = f"from ctxagent.jobs import status; print(status('{identifier}', 'job-tests')['state'])"
        out = subprocess.check_output([sys.executable, '-B', '-c', code])
        self.assertIn(b'running', out)
        self.assertTrue(jobs.cancel(identifier, 'job-tests')['cancel_requested'])
        self.assertEqual(self.wait(identifier)['state'], 'cancelled')
        self.assertIn(identifier, [j['id'] for j in jobs.list_owned('job-tests')])

    def test_status_metadata_read_retries_windows_sharing_violation_only(self):
        with patch.object(Path, 'read_text', side_effect=[PermissionError('sharing'), '{"state":"running"}']) as read:
            with patch('ctxagent.jobs.time.sleep'):
                self.assertEqual(jobs.read_json(Path('status.json'))['state'], 'running')
        self.assertEqual(read.call_count, 2)

    def test_log_cap_drains_and_reports(self):
        identifier = self.start_python("import sys\nsys.stdout.write('x' * (17 * 1024 * 1024))")
        data = self.wait(identifier)
        self.assertEqual(data['state'], 'succeeded', data)
        self.assertEqual(data['log_bytes']['stdout'], jobs.MAX_LOG_BYTES)
        self.assertEqual(data['dropped_bytes']['stdout'], 1024 * 1024)

    def test_cancellation_kills_descendant(self):
        import psutil
        marker = self.root / 'child.pid'
        source = ("import subprocess, sys, time\n"
                  "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                  f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
                  "time.sleep(60)\n")
        identifier = self.start_python(source)
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertTrue(marker.exists())
        child = psutil.Process(int(marker.read_text()))
        try:
            jobs.cancel(identifier, 'job-tests')
            self.assertEqual(self.wait(identifier)['state'], 'cancelled')
            self.assertFalse(child.is_running(), 'Descendant survived cancellation')
        finally:
            if child.is_running():
                child.kill()

    def test_atomic_metadata_replace_retries_windows_sharing_error(self):
        dest = self.root / 'status.json'
        dest.write_text('{"old": true}')
        original = jobs.os.replace
        attempts = []
        def replace(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError('simulated Windows sharing conflict')
            original(source, target)
        with patch.object(jobs.os, 'replace', replace), patch.object(jobs.time, 'sleep'):
            jobs.atomic_json(dest, {'state': 'running'})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(dest.read_text())['state'], 'running')
        with patch.object(jobs.os, 'replace', side_effect=PermissionError('permanent')), patch.object(jobs.time, 'sleep'):
            with self.assertRaises(PermissionError):
                jobs.atomic_json(dest, {'state': 'not-written'})
        self.assertEqual(json.loads(dest.read_text())['state'], 'running')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            jobs.start('', self.root, 'job-tests')
        with self.assertRaises(ValueError):
            jobs.start('echo bad', self.root, 'job-tests', 0)
        with self.assertRaises(ValueError):
            jobs.status('../escape', 'job-tests')



class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_artifact_authorization_and_busy_mutations(self):
        import httpx
        from PIL import Image
        from ctxagent.server import app
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'api.png'
            Image.new('RGB', (64, 64), 'blue').save(p)
            image = artifacts.view(p)[0]
            session = make_session()
            media = session.conv.new('user', K_MEDIA, 'evidence', images=[image])
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                with patch('ctxagent.server.agent.get_session', return_value=session):
                    response = await client.get(f"/api/chats/test/artifacts/{image['id']}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers['content-type'], 'image/png')
                    self.assertEqual((await client.get('/api/chats/test/artifacts/' + 'f' * 64)).status_code, 404)
                    await session.turn_lock.acquire()
                    try:
                        response = await client.patch('/api/chats/test/config', json={'config': {'temperature': 1}})
                        self.assertEqual(response.status_code, 409)
                        response = await client.post(f'/api/chats/test/segments/{media.id}/pin')
                        self.assertEqual(response.status_code, 409)
                    finally:
                        session.turn_lock.release()
                with patch('ctxagent.server.agent.get_session', return_value=make_session()):
                    self.assertEqual((await client.get(f"/api/chats/other/artifacts/{image['id']}")).status_code, 404)

    async def test_registered_tools_dispatch_and_permissions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test.png'
            Image.new('RGB', (32, 32)).save(p)
            session = make_session(root=td)
            for name in ('view_image', 'render_pdf', 'start_job', 'job_status', 'list_jobs', 'read_job_log', 'cancel_job'):
                self.assertIn(name, session.tool_names())
                self.assertTrue(tools.openai_schema([name])[0]['function']['parameters'])
            result = await session._exec_tool('view_image', {'path': str(p)})
            self.assertFalse(result.error)
            self.assertEqual(len(result.images), 1)
            result = await session._exec_tool('list_jobs', {})
            self.assertEqual(json.loads(result.text), [])
            session.cfg.enabled_tools = ['read_file']
            result = await session._exec_tool('view_image', {'path': str(p)})
            self.assertTrue(result.error)

if __name__ == '__main__':
    unittest.main(verbosity=2)
