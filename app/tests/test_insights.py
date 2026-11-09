"""Offline tests for the usage insights: the opt-in preference, what a turn records, and the
report built from the records. Scripted provider, temp data folder, no network."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_DATA = tempfile.TemporaryDirectory(prefix='ctxagent-insights-tests-')
os.environ.setdefault('CTXAGENT_DATA', _DATA.name)

from fastapi.testclient import TestClient  # noqa: E402

from ctxagent import agent, config, insights, insights_report, profiles, settings_api, store  # noqa: E402
from ctxagent.agent import AgentSession  # noqa: E402
from ctxagent.config import CompactionConfig, SessionConfig  # noqa: E402
from ctxagent.segments import Conversation  # noqa: E402
from tests.test_tool_call_failures import ScriptedProvider, call, run, text_turn, tool_turn  # noqa: E402


def usage(prompt=1000, read=800, out=50):
    return {'type': 'usage', 'usage': {'prompt_tokens': prompt, 'completion_tokens': out, 'cost': 0.001,
                                       'prompt_tokens_details': {'cached_tokens': read, 'cache_write_tokens': 0}}}


async def _price(*a, **k):
    return {'rates': {}, 'basis': 'test'}


class Isolated(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = patch.object(config, 'DATA_DIR', self.root / 'data'); p.start(); self.addCleanup(p.stop)
        p = patch.object(agent.accounting, 'pricing', _price); p.start(); self.addCleanup(p.stop)
        insights.refresh()
        self.addCleanup(insights.refresh)

    def session(self, workdir: Path, **cfg_kw) -> AgentSession:
        cfg = SessionConfig(compaction=CompactionConfig(enabled=True, rewrite_policy='agent'),
                            workdirs=[str(workdir)], model='anthropic/claude-opus-5.5', **cfg_kw)
        s = AgentSession('ins-' + os.urandom(4).hex(), cfg, Conversation())
        s.persist = lambda: None
        return s

    def events(self, chat_id):
        path = insights.folder() / f'{chat_id}.jsonl'
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class PreferenceTests(Isolated):
    def test_off_by_default_and_validated(self):
        self.assertIs(profiles.DEFAULT_PREFS['insights'], False)
        self.assertFalse(insights.enabled())
        self.assertIs(profiles.update_prefs({'insights': True})['insights'], True)
        insights.refresh()
        self.assertTrue(insights.enabled())
        with self.assertRaises(profiles.ProfileError):
            profiles.update_prefs({'insights': 'yes'})

    async def test_nothing_is_written_while_off(self):
        work = self.root / 'w'
        work.mkdir()
        (work / 'a.txt').write_text('hello', encoding='utf-8')
        s = self.session(work)
        provider = ScriptedProvider([usage()] + tool_turn(call('r1', 'read_file', {'path': 'a.txt'})),
                                    [usage()] + text_turn('ok'))
        with patch.object(agent.llm, 'stream_chat', provider):
            await run(s)
        self.assertFalse(insights.folder().exists())


class RecordingTests(Isolated):
    async def asyncSetUp(self):
        p = patch.object(insights, 'enabled', lambda: True); p.start(); self.addCleanup(p.stop)
        self.work = self.root / 'w'
        self.work.mkdir()
        (self.work / 'a.txt').write_text('hello world\n', encoding='utf-8')
        self.s = self.session(self.work)
        provider = ScriptedProvider(
            [usage(1000)] + tool_turn(call('r1', 'read_file', {'path': 'a.txt'}), text='Reading the file.'),
            [usage(1200)] + tool_turn(call('e1', 'edit_file', {'path': 'a.txt', 'old_string': 'nothere', 'new_string': 'x'})),
            [usage(1300)] + tool_turn(call('e2', 'edit_file', {'path': 'a.txt', 'old_string': 'hello', 'new_string': 'bye'})),
            [usage(1400)] + text_turn('All done: a.txt now says bye.'),
            [usage(1500)] + tool_turn(call('c1', 'compact_context', {'summary': 'Edited a.txt: hello -> bye.',
                                                                     'dry_run': True}),
                                      text='The edit is done; checking what a hand-off would save.'),
            [usage(1600)] + text_turn('Nothing to hand off yet.'),
        )
        with patch.object(agent.llm, 'stream_chat', provider):
            await run(self.s, 'Replace hello by bye in a.txt <b>now</b>')
            await run(self.s, 'Is a hand-off worth it?')
        self.ev = self.events(self.s.chat_id)

    def test_turn_requests_tools_and_context(self):
        kinds = [e['kind'] for e in self.ev]
        self.assertEqual(kinds[0], 'turn_start')
        self.assertEqual(kinds[-1], 'turn_end')
        self.assertEqual((kinds.count('turn_start'), kinds.count('turn_end')), (2, 2))
        self.assertEqual(kinds.count('request'), 6)
        self.assertEqual(kinds.count('tool'), 4)
        start = self.ev[0]
        self.assertEqual((start['turn'], start['model'], start['self_steered']), (1, 'anthropic/claude-opus-5.5', True))
        self.assertIn('Replace hello by bye', start['user'])
        req = [e for e in self.ev if e['kind'] == 'request']
        self.assertEqual([r['prompt_tokens'] for r in req], [1000, 1200, 1300, 1400, 1500, 1600])
        self.assertEqual((req[0]['cache_read'], req[0]['uncached'], req[0]['tool_calls']), (800, 200, 1))
        tools = [e for e in self.ev if e['kind'] == 'tool']
        self.assertEqual([t['name'] for t in tools], ['read_file', 'edit_file', 'edit_file', 'compact_context'])
        self.assertEqual(tools[0]['category'], 'inspect')
        self.assertTrue(tools[0]['ok'] and tools[0]['segment'])
        failed = tools[1]
        self.assertEqual((failed['ok'], failed['error_class'], failed['category']), (False, 'result.edit_not_found', 'edit'))
        # Edit strings are recorded as sizes, never as text.
        self.assertEqual(failed['args']['old_string'], {'chars': 7, 'lines': 1})
        self.assertEqual(failed['args']['path'], 'a.txt')
        self.assertTrue(tools[2]['ok'] and tools[2]['mutates'])
        ctx = [e for e in self.ev if e['kind'] == 'context']
        self.assertEqual(len(ctx), 1)
        c = ctx[0]
        self.assertEqual((c['source'], c['dry_run'], c['applied'], c['turn']), ('compact_context', True, False, 2))
        self.assertIn('Is a hand-off worth it?', c['task'])
        self.assertIn('a.txt now says bye', c['last_answer'])
        self.assertIn('checking what a hand-off would save', c['call_text'])
        self.assertIn('Edited a.txt', c['summary'])
        self.assertEqual(c['since']['batches'], 3)
        first_end = [e for e in self.ev if e['kind'] == 'turn_end'][0]
        self.assertEqual((first_end['turn'], first_end['outcome'], first_end['tools'], first_end['tool_errors'],
                          first_end['requests']), (1, 'final', 3, 1, 4))
        self.assertIn('a.txt now says bye', first_end['answer'])
        self.assertEqual(self.ev[-1]['turn'], 2)

    def test_report(self):
        stats = insights_report.analyse(insights_report.load())
        self.assertEqual(stats['overview']['tool_calls'], 4)
        self.assertEqual(stats['overview']['tool_failures'], 1)
        e = stats['edits']
        self.assertEqual((e['calls'], e['recovered'], e['unrecovered'], e['calls_to_fix_median']), (2, 1, 0, 1))
        self.assertEqual(e['read_first']['read'], {'calls': 2, 'failed': 1})
        self.assertEqual(e['classes'], [('result.edit_not_found', 1)])
        self.assertEqual(stats['inspection']['read_file_calls'], 1)
        self.assertEqual(stats['context']['by_source']['compact_context']['dry_run'], 1)
        chat = stats['chats'][0]
        self.assertEqual((len(chat['requests']), chat['peak'], chat['turns']), (6, 1600, 2))
        page = insights_report.render_html(stats)
        self.assertIn('<svg', page)
        self.assertIn('Edited a.txt: hello -&gt; bye.', page)
        self.assertNotIn('hello -> bye', page)
        md = insights_report.render_markdown(stats)
        self.assertIn('| edit_file | edit | 2 | 1 |', md)


class ReportTests(unittest.TestCase):
    def test_dropped_results_hints_and_difficult_tools(self):
        t = 1000.0
        evs = [{'kind': 'turn_start', 'ts': t, 'turn': 1, 'model': 'm', 'window': 1000}]
        for i in range(6):
            evs.append({'kind': 'request', 'ts': t + i, 'turn': 1, 'model': 'm', 'prompt_tokens': 100 * (i + 1)})
            evs.append({'kind': 'tool', 'ts': t + i + 0.1, 'turn': 1, 'name': 'grep_search', 'category': 'search',
                        'ok': i % 2 == 0, 'error_class': None if i % 2 == 0 else 'result.bad_regex',
                        'error': None if i % 2 == 0 else 'Error: Bad regex: x', 'tokens': 50,
                        'segment': f'aaaaaa00000{i}', 'empty': i == 0})
        evs.append({'kind': 'context', 'ts': t + 6.1, 'turn': 1, 'source': 'hint', 'applied': False})
        evs.append({'kind': 'context', 'ts': t + 6.2, 'turn': 1, 'source': 'curate_context', 'applied': True,
                    'before': 600, 'after': 500, 'freed': 100, 'dropped': ['aaaaaa000000 -50 tok', 'aaaaaa000002 -50 tok']})
        stats = insights_report.analyse({'c1': evs})
        grep = stats['tools'][0]
        self.assertEqual((grep['tool'], grep['calls'], grep['failed'], grep['dropped_later']), ('grep_search', 6, 3, 2))
        self.assertEqual(stats['context']['hints'], 1)
        self.assertEqual(stats['context']['hints_followed'], 1)
        self.assertEqual(stats['inspection']['empty_searches'], 1)
        self.assertEqual(stats['difficult']['m'][0]['tool'], 'grep_search')
        self.assertEqual(stats['difficult']['m'][0]['classes'], [('result.bad_regex', 3)])
        self.assertIn('window', insights_report.render_html(stats))

    def test_commands(self):
        evs = [{'kind': 'tool', 'ts': 1.0 + i, 'name': 'run_command', 'category': 'shell', 'ok': True, 'exit': code,
                'error_class': 'info.command_nonzero_exit' if code else None, 'args': {'command': cmd}, 'tail': tail}
               for i, (cmd, code, tail) in enumerate([
                   ('& C:\Python\python.exe -m pytest', 1, '... 3 failed, 10 passed in 2s'),
                   ('python -c "import nope"', 1, "ModuleNotFoundError: No module named 'nope'"),
                   ('git status; ls', 0, ''),
                   ("Get-Content a.txt | head -5", 1, "head : The term 'head' is not recognized as a cmdlet"),
               ])]
        c = insights_report.analyse({'c': evs})['commands']
        self.assertEqual((c['calls'], c['nonzero']), (4, 3))
        self.assertEqual(dict((h, (n, z)) for h, n, z in c['heads']), {'python': (2, 2), 'git': (1, 0), 'get-content': (1, 1)})
        self.assertEqual(dict(c['signatures']), {'tests failed': 1, 'Python: module not found': 1,
                                                 'PowerShell: not a command': 1})

    def test_command_program(self):
        cases = {
            "cd C:\\x; $env:A='1'; @'\nimport json\nprint(1)\n'@ | python -": 'python',
            "$c = @'\nprint(1)\n'@\n$c | & C:\\py\\python.exe -": 'python',
            'cd /work && python -m pytest -q': 'python',
            '& "C:\\Program Files\\Godot\\godot.exe" --headless --script t.gd': 'godot',
            'for ($i=0; $i -lt 3; $i++) { Start-Sleep 1 }': 'powershell script',
            'function F($a) { $a }; F 1': 'powershell script',
            "Get-Content a.txt | Select-String 'x'": 'get-content',
            'git status --short': 'git',
        }
        for command, program in cases.items():
            self.assertEqual(insights.command_program(command), program, command)

    def test_empty_report(self):
        stats = insights_report.analyse({})
        self.assertIn('No insights recorded yet', insights_report.render_html(stats))
        self.assertIn('# Usage insights', insights_report.render_markdown(stats))


class ApiTests(Isolated):
    def test_summary_and_report_endpoints(self):
        with patch.object(store, 'DB_PATH', self.root / 'test.db'):
            store.init()
            from ctxagent.server import app
            client = TestClient(app, base_url='http://127.0.0.1:8770')
            r = client.get('/api/settings/insights')
            self.assertEqual(r.status_code, 200)
            self.assertEqual((r.json()['enabled'], r.json()['chats']), (False, 0))
            self.assertEqual(client.patch('/api/settings/prefs', json={'prefs': {'insights': True}}).json()['prefs']['insights'], True)
            insights.write('chat1', 'turn_start', turn=1, model='m', user='hi')
            opened = []
            with patch.object(settings_api, '_open_in_file_manager', opened.append):
                r = client.post('/api/settings/insights/report', json={'open': True})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()['chats'], 1)
                self.assertEqual(opened, [Path(r.json()['html'])])
                self.assertTrue((insights.folder() / 'report.md').is_file())
                self.assertEqual(client.post('/api/settings/open-folder', json={'which': 'insights'}).status_code, 200)


if __name__ == '__main__':
    unittest.main()
