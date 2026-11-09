"""Regression tests for the tool-call / turn failure classes found in recorded sessions.

Every fixture is synthetic: it reproduces the SHAPE of a recorded failure (segment kinds,
call/result order, provider events), never recorded content. See TOOL_CALL_FAILURES.md for
the class table and tools_scripts/scan_tool_failures.py for the scanner that found them.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

_DATA = tempfile.TemporaryDirectory(prefix='ctxagent-toolfail-tests-')
os.environ.setdefault('CTXAGENT_DATA', _DATA.name)

import httpx  # noqa: E402

from ctxagent import agent, llm, tools, tool_batch  # noqa: E402
from ctxagent.agent import AgentSession  # noqa: E402
from ctxagent.config import CompactionConfig, SessionConfig  # noqa: E402
from ctxagent.segments import (Conversation, Segment, K_SYSTEM, K_USER, K_TOOL_CALL,  # noqa: E402
                               K_TOOL_RESULT, K_ASSISTANT, INTERRUPTED_RESULT, wire_problems)


# --------------------------------------------------------------------------- helpers
def call(cid, name, args=None, raw=None):
    return {'id': cid, 'type': 'function',
            'function': {'name': name, 'arguments': raw if raw is not None else json.dumps(args or {})}}


def make_session(root=None, **cfg_kw):
    cfg = SessionConfig(compaction=CompactionConfig(enabled=False),
                        workdirs=[str(root)] if root else [], **cfg_kw)
    session = AgentSession('toolfail-' + os.urandom(4).hex(), cfg, Conversation())
    session.persist = lambda: None
    return session


class ScriptedProvider:
    """Stand-in for llm.stream_chat. Each request takes the next script entry: a list of
    events, an exception instance (raised before anything is yielded) or a callable
    (kwargs -> list of events). Every request's messages are checked against the provider
    pairing rules, which is the regression check for the whole pairing class."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.requests = []
        self.problems = []

    async def __call__(self, **kw):
        self.requests.append(kw)
        self.problems.extend(wire_problems(kw['messages']))
        step = self.scripts.pop(0) if self.scripts else [{'type': 'text', 'text': 'done'},
                                                           {'type': 'done', 'finish_reason': 'stop'}]
        if callable(step):
            step = step(kw)
        if isinstance(step, BaseException):
            raise step
        for ev in step:
            if isinstance(ev, BaseException):
                raise ev
            yield ev


def tool_turn(*calls, text=''):
    evs = [{'type': 'text', 'text': text}] if text else []
    return evs + [{'type': 'tool_calls', 'tool_calls': list(calls)}, {'type': 'done', 'finish_reason': 'tool_calls'}]


def text_turn(text='finished', finish='stop'):
    return [{'type': 'text', 'text': text}, {'type': 'done', 'finish_reason': finish}]


async def run(session, prompt='go'):
    return [ev async for ev in session.run_turn(prompt)]


def kinds(conv):
    return [s.kind for s in conv.segments]


# --------------------------------------------------------------------------- pairing
class PairingTests(unittest.IsolatedAsyncioTestCase):
    """Class: a segment written between a call and its result (document_close's handoff note)
    made the end-of-turn repair add an `[interrupted]` placeholder for a call that HAD a result.
    Recorded: 16 placeholders in one live chat, after which Anthropic rejected every request
    ("each tool_use must have a single result. Found multiple `tool_result` blocks")."""

    def recorded_shape(self, *, broken):
        """[call(read, read, run, close, close)] r1 r2 r3 [note] r4 [note] r5, optionally with
        the two placeholders the old repair inserted after r3 (the stored broken state)."""
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys')
        c.new('user', K_USER, 'task')
        ids = ['c1', 'c2', 'c3', 'c4', 'c5']
        names = ['read_job_log', 'read_job_log', 'run_command', 'document_close', 'document_close']
        c.new('assistant', K_TOOL_CALL, '', tool_calls=[call(i, n) for i, n in zip(ids, names)])
        for i in ids[:3]:
            c.new('tool', K_TOOL_RESULT, f'result {i}', tool_call_id=i, tool_name='x')
        if broken:
            for i in ids[3:]:
                c.new('tool', K_TOOL_RESULT, INTERRUPTED_RESULT, tool_call_id=i, tool_name='document_close')
        c.new('user', K_USER, 'Document handoff for a.py (close note, not file evidence):\nnote A')
        c.new('tool', K_TOOL_RESULT, 'Closed a.py', tool_call_id='c4', tool_name='document_close')
        c.new('user', K_USER, 'Document handoff for b.py (close note, not file evidence):\nnote B')
        c.new('tool', K_TOOL_RESULT, 'Closed b.py', tool_call_id='c5', tool_name='document_close')
        c.new('assistant', K_ASSISTANT, 'ok')
        return c

    def test_repair_regroups_interleaved_results_without_placeholders(self):
        c = self.recorded_shape(broken=False)
        self.assertTrue(c.repair_tool_pairs())
        bodies = [s.body for s in c.segments]
        self.assertNotIn(INTERRUPTED_RESULT, bodies)
        call_at = kinds(c).index(K_TOOL_CALL)
        self.assertEqual([s.tool_call_id for s in c.segments[call_at + 1:call_at + 6]], ['c1', 'c2', 'c3', 'c4', 'c5'])
        self.assertTrue(c.segments[call_at + 6].body.startswith('Document handoff for a.py'))
        self.assertTrue(c.segments[call_at + 7].body.startswith('Document handoff for b.py'))
        self.assertEqual(wire_problems(c.to_messages()), [])
        self.assertEqual(c.repair_tool_pairs(), 0)  # idempotent

    def test_repair_heals_stored_broken_chat_and_keeps_cached_prefix(self):
        c = self.recorded_shape(broken=True)
        c.freeze_prefix(len(c.segments))  # the recorded chat had the whole batch frozen
        self.assertTrue(c.repair_tool_pairs())
        self.assertEqual(sum(s.body == INTERRUPTED_RESULT for s in c.segments), 0)
        self.assertEqual(sum(s.tool_call_id == 'c4' for s in c.segments), 1)
        self.assertEqual(wire_problems(c.to_messages()), [])
        # Only the region from the first changed position leaves the cached prefix.
        first_result = kinds(c).index(K_TOOL_CALL) + 4
        self.assertEqual(c.frozen_upto, first_result)
        self.assertTrue(all(s.frozen for s in c.segments[:first_result]))

    def test_to_messages_is_valid_even_for_unrepaired_shapes(self):
        """Delegated chats are loaded by the server without repair; the request must still be
        well-formed: results right after their call, one per id, a placeholder for a call
        without any, and JSON-object arguments."""
        for broken in (False, True):
            c = self.recorded_shape(broken=broken)
            msgs = c.to_messages()
            self.assertEqual(wire_problems(msgs), [], broken)
            roles = [m['role'] for m in msgs]
            at = roles.index('assistant')
            self.assertEqual(roles[at:at + 8], ['assistant'] + ['tool'] * 5 + ['user', 'user'])
            self.assertEqual([m['content'] for m in msgs[at + 4:at + 6]], ['Closed a.py', 'Closed b.py'])
        c = Conversation()
        c.new('user', K_USER, 'task')
        c.new('assistant', K_TOOL_CALL, '', tool_calls=[call('t1', 'write_file', raw='{"path": "a.py", "content": "x')])
        c.new('user', K_USER, 'next question')
        msgs = c.to_messages()
        self.assertEqual(wire_problems(msgs), [])
        self.assertEqual(json.loads(msgs[1]['tool_calls'][0]['function']['arguments']),
                         {'_unparsed_arguments': '{"path": "a.py", "content": "x'})
        self.assertIn('[interrupted]', msgs[2]['content'])

    def test_orphan_result_still_becomes_a_note(self):
        c = Conversation()
        c.new('user', K_USER, 'task')
        c.new('tool', K_TOOL_RESULT, 'lost call', tool_call_id='gone', tool_name='read_file')
        msgs = c.to_messages()
        self.assertEqual([m['role'] for m in msgs], ['user', 'user'])
        self.assertEqual(wire_problems(msgs), [])

    async def test_document_close_handoff_end_to_end(self):
        """The real tool and loop: close with a handoff note inside a batch, then two more
        requests. No placeholder, the note after the results, every request well-formed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.py').write_text('print(1)\n', encoding='utf-8')
            s = make_session(root)
            s.cfg.enabled_tools = list(s.cfg.enabled_tools) + ['document_close', 'document_open', 'read_file']
            doc = s.conv.workspace.open(s.sandbox(), str(root / 'a.py'))
            provider = ScriptedProvider(
                tool_turn(call('k1', 'read_file', {'path': 'a.py'}),
                          call('k2', 'document_close', {'document_id': doc, 'summary': 'a.py prints 1'}),
                          call('k3', 'read_file', {'path': 'a.py'})),
                text_turn('closed'))
            with patch.object(agent.llm, 'stream_chat', provider):
                await run(s, 'close it')
                await run(s, 'next')
            self.assertEqual(provider.problems, [])
            self.assertFalse(any(x.body == INTERRUPTED_RESULT for x in s.conv.segments))
            at = kinds(s.conv).index(K_TOOL_CALL)
            self.assertEqual([x.tool_call_id for x in s.conv.segments[at + 1:at + 4]], ['k1', 'k2', 'k3'])
            self.assertTrue(s.conv.segments[at + 4].body.startswith('Document handoff for'))


# --------------------------------------------------------------------------- transient provider failures
STANDARD = [{'tier': 'default', 'provider': None, 'notice': None, 'mode': 'standard'}]


def sse(*chunks, done=True):
    return ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + ('data: [DONE]\n\n' if done else '')


def text_chunk(text):
    return {'id': 'gen-1', 'model': 'm', 'choices': [{'delta': {'content': text}}]}


FINISH = {'choices': [{'delta': {}, 'finish_reason': 'stop'}],
          'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'cost': 0.0001}}


class CutStream(httpx.AsyncByteStream):
    """SSE bytes, then the connection dies (fault injection for mid-stream cuts)."""

    def __init__(self, text, exc):
        self.text, self.exc = text, exc

    async def __aiter__(self):
        yield self.text.encode()
        raise self.exc


class StreamRetryTests(unittest.IsolatedAsyncioTestCase):
    """Class: a provider/network failure ended the turn. Recorded: 25 upstream transport
    errors (connection reset by peer, WinError 64/121, DNS), 19 in-band provider errors
    ('Stream ended before a terminal response event', 'internal server error', 'overloaded')
    and 8 HTTP 5xx; each one lost the rest of its turn. Failures BEFORE any output are now
    retried transparently; after output started nothing is replayed."""

    def setUp(self):
        self.sleeps = []

        async def no_sleep(seconds):
            self.sleeps.append(seconds)
        self.enterContext(patch.object(llm, '_sleep', no_sleep))
        self.enterContext(patch.object(llm, '_headers', return_value={}))

        async def plans(*a, **k):
            return [dict(p) for p in STANDARD]
        self.enterContext(patch.object(llm.flex, 'plans', plans))
        self.sent = []

    def serve(self, *responses):
        """Each request gets the next response; an exception instance is raised instead."""
        queue = list(responses)

        def handler(request):
            self.sent.append(json.loads(request.content))
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        real = httpx._client.AsyncClient  # the class itself, even when already patched
        self.enterContext(patch.object(llm.httpx, 'AsyncClient',
                                       lambda **kw: real(transport=httpx.MockTransport(handler), **kw)))

    async def stream(self, **kw):
        return [e async for e in llm.stream_chat(model='openai/gpt-test', messages=[{'role': 'user', 'content': 'hi'}],
                                                 **{'max_tokens': 32000, **kw})]

    def ok(self):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=sse(text_chunk('hello'), FINISH))

    async def test_connection_reset_before_output_is_retried(self):
        self.serve(httpx.ConnectError('[Errno 104] Connection reset by peer'), self.ok())
        events = await self.stream()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.sent[0], self.sent[1])
        self.assertFalse([e for e in events if e['type'] == 'error'])
        self.assertTrue(any(e['type'] == 'notice' and 'retrying' in e['text'] for e in events))
        self.assertEqual(''.join(e['text'] for e in events if e['type'] == 'text'), 'hello')
        self.assertEqual(self.sleeps, [2.0])

    async def test_http_5xx_and_in_band_provider_errors_are_retried(self):
        unavailable = {'error': {'code': 502, 'message': 'Stream ended before a terminal response event',
                                 'metadata': {'error_type': 'provider_unavailable'}}}
        self.serve(httpx.Response(502, text='{"error":{"message":"Provider returned error","code":502}}'),
                   httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=sse(unavailable, done=False)),
                   httpx.Response(200, json={'error': {'code': 500, 'message': 'internal server error'}}),
                   self.ok())
        events = await self.stream()
        self.assertEqual(len(self.sent), 4)
        self.assertEqual(events[-1], {'type': 'done', 'finish_reason': 'stop'})

    async def test_retries_are_bounded(self):
        self.serve(*[httpx.Response(503, text='busy')] * (llm.STREAM_RETRIES + 1))
        events = await self.stream()
        self.assertEqual(len(self.sent), llm.STREAM_RETRIES + 1)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('interrupted', events[-1])

    async def test_cut_mid_tool_call_is_never_executed_or_replayed(self):
        partial = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'c1', 'function': {
            'name': 'write_file', 'arguments': '{"path": "a.py", "content": "par'}}]}}]}
        self.serve(httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  stream=CutStream(sse(partial, done=False),
                                                   httpx.RemoteProtocolError('peer closed connection'))))
        events = await self.stream()
        self.assertEqual(len(self.sent), 1)
        self.assertFalse([e for e in events if e['type'] == 'tool_calls'])
        self.assertEqual(events[-1]['type'], 'error')
        self.assertTrue(events[-1]['interrupted'])

    async def test_clean_close_without_finish_is_incomplete_not_done(self):
        self.serve(httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text=sse(text_chunk('half an ans'), done=False)))
        events = await self.stream()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertTrue(events[-1]['interrupted'])
        self.assertFalse(any(e['type'] == 'done' for e in events))

    async def test_402_affordable_max_tokens_retried_once(self):
        body = ('{"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up '
                'to 32000 tokens, but can only afford 20000.","code":402}}')
        self.serve(httpx.Response(402, text=body), self.ok())
        events = await self.stream()
        self.assertEqual([r['max_tokens'] for r in self.sent], [32000, 19936])
        self.assertTrue(any(e['type'] == 'notice' and 'credits' in e['text'] for e in events))
        self.assertEqual(events[-1]['type'], 'done')
        self.sent.clear()
        self.serve(httpx.Response(402, text=body.replace('20000', '900')))
        events = await self.stream()
        self.assertEqual(len(self.sent), 1)  # too little left to be useful: plain error
        self.assertEqual(events[-1]['type'], 'error')

    async def test_missing_tool_call_ids_get_unique_ids(self):
        calls = {'choices': [{'delta': {'tool_calls': [
            {'index': 0, 'function': {'name': 'read_file', 'arguments': '{"path": "a"}'}},
            {'index': 1, 'function': {'name': 'read_file', 'arguments': '{"path": "b"}'}}]}}]}
        self.serve(httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text=sse(calls, {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})))
        events = await self.stream()
        ids = [c['id'] for c in next(e for e in events if e['type'] == 'tool_calls')['tool_calls']]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(i.startswith('call_') and len(i) > 10 for i in ids))


class StreamRecoveryLoopTests(unittest.IsolatedAsyncioTestCase):
    """The agent side of the same class: a response cut off after it started keeps the text
    the user saw and continues (bounded); a half-streamed tool call is never run."""

    async def test_cut_off_text_is_kept_and_the_turn_continues(self):
        s = make_session()
        provider = ScriptedProvider(
            [{'type': 'text', 'text': 'Working on it'},
             {'type': 'error', 'error': "{'code': 502, 'message': 'Stream ended before a terminal response event'}",
              'interrupted': True}],
            text_turn('All done.'))
        with patch.object(agent.llm, 'stream_chat', provider):
            events = await run(s)
        self.assertEqual(len(provider.requests), 2)
        self.assertFalse([e for e in events if e['type'] == 'error'])
        bodies = [x.body for x in s.conv.segments]
        self.assertTrue(any(b.startswith('Working on it') and 'cut off' in b for b in bodies))
        self.assertTrue(any(b.startswith('[harness note] Your previous response was cut off') for b in bodies))
        self.assertEqual(events[-2]['type'], 'final')
        self.assertEqual(provider.problems, [])

    async def test_recovery_is_bounded_and_failure_keeps_partial_text(self):
        s = make_session()
        cut = [{'type': 'text', 'text': 'partial'}, {'type': 'error', 'error': 'ReadError: reset', 'interrupted': True}]
        provider = ScriptedProvider(cut, cut, cut)
        with patch.object(agent.llm, 'stream_chat', provider):
            events = await run(s)
        self.assertEqual(len(provider.requests), agent.STREAM_RECOVERIES + 1)
        self.assertEqual([e for e in events if e['type'] == 'error'][-1]['error'], 'ReadError: reset')
        self.assertTrue(s.conv.segments[-1].body.startswith('partial\n\n[request failed]'))

    async def test_permanent_error_is_not_continued(self):
        s = make_session()
        provider = ScriptedProvider([{'type': 'error', 'error': 'HTTP 400: bad request'}])
        with patch.object(agent.llm, 'stream_chat', provider):
            events = await run(s)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(s.conv.segments[-1].body, '[request failed] HTTP 400: bad request')


# --------------------------------------------------------------------------- empty / length-capped responses
EMPTY_AT_LIMIT = [{'type': 'usage', 'usage': {'prompt_tokens': 36000, 'completion_tokens': 32000}},
                  {'type': 'done', 'finish_reason': 'length'}]


class EmptyResponseTests(unittest.IsolatedAsyncioTestCase):
    """Class: the model reasons to the output limit without text or a tool call. Recorded on
    hard puzzle tasks (Opus/Sonnet/Haiku/Fable, effort high, 32k): 24 harness notes, 15 turns
    stopped after three identical attempts of ~350 s and 32000 reasoning tokens each,
    because every retry ran at the same effort and restarted its reasoning from scratch."""

    def efforts(self, provider):
        return [r['reasoning_effort'] for r in provider.requests]

    async def test_retries_step_the_effort_down(self):
        s = make_session(reasoning_effort='high', max_output_tokens=32000)
        provider = ScriptedProvider(EMPTY_AT_LIMIT, EMPTY_AT_LIMIT, text_turn('ANSWER: 42'))
        with patch.object(agent.llm, 'stream_chat', provider), patch.object(s, '_account_usage', self._no_usage):
            events = await run(s)
        self.assertEqual(self.efforts(provider), ['high', 'medium', 'low'])
        notes = [x.body for x in s.conv.segments if x.body.startswith('[harness note]')]
        self.assertIn("reasoning effort 'medium'", notes[0])
        self.assertIn("reasoning effort 'low'", notes[1])
        self.assertIn('private reasoning was lost', notes[0])
        self.assertEqual(events[-2]['type'], 'final')
        self.assertEqual(s.cfg.reasoning_effort, 'high')  # the setting itself is untouched

    async def test_three_empty_responses_still_stop(self):
        s = make_session(reasoning_effort='high')
        provider = ScriptedProvider(EMPTY_AT_LIMIT, EMPTY_AT_LIMIT, EMPTY_AT_LIMIT)
        with patch.object(agent.llm, 'stream_chat', provider), patch.object(s, '_account_usage', self._no_usage):
            events = await run(s)
        self.assertEqual(len(provider.requests), 3)
        self.assertTrue(s.conv.segments[-1].body.startswith('[stopped]'))
        self.assertEqual([e for e in events if e['type'] == 'error'][-1]['error'], s.conv.segments[-1].body)

    async def test_effort_returns_to_the_setting_after_a_productive_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'a.txt').write_text('x', encoding='utf-8')
            s = make_session(Path(tmp), reasoning_effort='high')
            provider = ScriptedProvider(EMPTY_AT_LIMIT, tool_turn(call('r1', 'read_file', {'path': 'a.txt'})),
                                        EMPTY_AT_LIMIT, text_turn('ok'))
            with patch.object(agent.llm, 'stream_chat', provider), patch.object(s, '_account_usage', self._no_usage):
                await run(s)
        # The empty count restarts after the tool call: the second empty is a first retry again.
        self.assertEqual(self.efforts(provider), ['high', 'medium', 'high', 'medium'])

    async def test_uncapped_empty_response_keeps_the_effort(self):
        s = make_session(reasoning_effort='high')
        provider = ScriptedProvider([{'type': 'done', 'finish_reason': 'stop'}], text_turn('ok'))
        with patch.object(agent.llm, 'stream_chat', provider):
            await run(s)
        self.assertEqual(self.efforts(provider), ['high', 'high'])

    async def test_xhigh_steps_down_to_high_then_low(self):
        s = make_session(reasoning_effort='xhigh', max_output_tokens=32000)
        provider = ScriptedProvider(EMPTY_AT_LIMIT, EMPTY_AT_LIMIT, text_turn('ANSWER: 42'))
        with patch.object(agent.llm, 'stream_chat', provider), patch.object(s, '_account_usage', self._no_usage):
            await run(s)
        self.assertEqual(self.efforts(provider), ['xhigh', 'high', 'low'])

    def test_openrouter_reasoning_payload(self):
        # xhigh and max go out as they are; OpenRouter maps them down where a model stops lower.
        self.assertEqual(llm.reasoning_payload('xhigh'), {'effort': 'xhigh'})
        self.assertEqual(llm.reasoning_payload('max'), {'effort': 'max'})
        self.assertEqual(llm.reasoning_payload('off'), {'enabled': False})
        self.assertIsNone(llm.reasoning_payload('default'))
        self.assertIsNone(llm.reasoning_payload(''))
        self.assertEqual(llm._reasoning_ladder('xhigh', 'google/gemini-4-argon'),
                         [{'effort': 'xhigh'}, {'effort': 'low'}, None])

    def test_lower_effort_ladder(self):
        self.assertEqual([agent.lower_effort(e, 1) for e in ('max', 'xhigh', 'high', 'medium', 'default')],
                         ['high', 'high', 'medium', 'low', 'medium'])
        self.assertEqual(agent.lower_effort('high', 2), 'low')
        self.assertIsNone(agent.lower_effort('low', 1))
        self.assertIsNone(agent.lower_effort('off', 1))

    async def _no_usage(self, *a, **k):
        return None


# --------------------------------------------------------------------------- hanging shell commands
def py_command(script: Path) -> str:
    if sys.platform == 'win32':
        return f"& '{sys.executable}' '{script}'"
    import shlex
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


GRANDCHILD = r'''
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(%d)"], stdout=sys.stdout, stderr=sys.stderr)
open(sys.argv[0] + ".pid", "w").write(str(child.pid))
print("phase1", flush=True)
time.sleep(%d)
'''


def _proc_alive(pid):
    """True while `pid` runs; a process that exits mid-check counts as gone."""
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class HangingCommandTests(unittest.IsolatedAsyncioTestCase):
    """Class: a turn hung on run_command. Recorded: one call 'timed out after 60s' and returned
    after 3753 s (a grandchild kept the output pipe open, and subprocess.run waits for EOF
    again after killing only the shell); one call passed timeout=60000 (milliseconds) and
    held its turn until the 3600 s cap."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sb = tools.Sandbox([str(self.root)])

    def script(self, name, body):
        p = self.root / name
        p.write_text(body, encoding='utf-8')
        return p

    def grandchild_pid(self, script):
        import psutil
        pid = int(Path(str(script) + '.pid').read_text())
        self.addCleanup(lambda: [p.kill() for p in psutil.process_iter() if p.pid == pid])
        return pid

    async def test_python_output_outside_the_ansi_code_page(self):
        # Recorded: a script printing non-ASCII text crashed with UnicodeEncodeError in cp1252.py,
        # because Python writes to a pipe in the ANSI code page on Windows; the result is read as UTF-8.
        p = self.script('say.py', 'print("\\u2713 \\u00dcbung \\u2192 fertig")\n')
        with patch.dict(os.environ):
            os.environ.pop('PYTHONIOENCODING', None)
            os.environ.pop('PYTHONUTF8', None)
            r = await tools.t_run_command(self.sb, py_command(p), timeout=60)
        self.assertFalse(r.error, r.text)
        self.assertIn('✓ Übung → fertig', r.text)

    async def test_stdin_is_closed(self):
        p = self.script('ask.py', 'x = input("name? ")\nprint("got", x)\n')
        t0 = time.monotonic()
        r = await tools.t_run_command(self.sb, py_command(p), timeout=60)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertIn('EOFError', r.text)
        self.assertTrue(r.error)

    async def test_background_process_holding_the_pipe_does_not_hang(self):
        p = self.script('bg.py', GRANDCHILD % (40, 0))
        t0 = time.monotonic()
        r = await tools.t_run_command(self.sb, py_command(p), timeout=30)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertIn('phase1', r.text)
        self.assertIn('still running and holds its output', r.text)
        self.grandchild_pid(p)

    async def test_timeout_kills_the_tree_and_keeps_partial_output(self):
        import psutil
        p = self.script('slow.py', GRANDCHILD % (60, 60))
        t0 = time.monotonic()
        r = await tools.t_run_command(self.sb, py_command(p), timeout=3)
        self.assertLess(time.monotonic() - t0, 25)
        self.assertTrue(r.text.startswith('Error: Command timed out after 3s'))
        self.assertIn('phase1', r.text)
        self.assertIn('start_job', r.text)
        self.assertTrue(r.error)
        pid = self.grandchild_pid(p)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        alive = _proc_alive(pid)
        self.assertFalse(alive)

    async def test_cancel_kills_the_command(self):
        import psutil
        p = self.script('long.py', GRANDCHILD % (60, 60))
        task = asyncio.create_task(tools.t_run_command(self.sb, py_command(p), timeout=120))
        deadline = time.monotonic() + 20
        while not Path(str(p) + '.pid').exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        pid = self.grandchild_pid(p)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 10
        while _proc_alive(pid) \
                and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        self.assertFalse(_proc_alive(pid))

    async def test_stop_kills_a_running_command_and_keeps_its_output(self):
        import psutil
        p = self.script('stop.py', GRANDCHILD % (60, 60))
        stop = asyncio.Event()
        token = tools.RUN_CANCEL.set(stop)  # what the tool batch does with the chat's Stop event
        try:
            t0 = time.monotonic()
            task = asyncio.create_task(tools.t_run_command(self.sb, py_command(p), timeout=120))
            while not Path(str(p) + '.pid').exists() and time.monotonic() - t0 < 20:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)
            stop.set()
            r = await task
        finally:
            tools.RUN_CANCEL.reset(token)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertTrue(r.text.startswith('Command stopped by the user'))
        self.assertIn('phase1', r.text)
        pid = self.grandchild_pid(p)
        deadline = time.monotonic() + 5
        while _proc_alive(pid) \
                and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        self.assertFalse(_proc_alive(pid))

    def test_timeout_argument(self):
        self.assertEqual(tools.command_timeout(60000)[0], 60)
        self.assertIn('milliseconds', tools.command_timeout(60000)[1])
        self.assertEqual(tools.command_timeout(7200), (600, f'timeout capped at 600 s; use start_job for longer work'))
        self.assertEqual(tools.command_timeout('30'), (30, ''))
        self.assertEqual(tools.command_timeout(0)[0], tools.RUN_COMMAND_DEFAULT_TIMEOUT)
        self.assertEqual(tools.command_timeout('soon')[0], tools.RUN_COMMAND_DEFAULT_TIMEOUT)

    async def test_ordinary_commands_unchanged(self):
        p = self.script('ok.py', 'import sys\nprint("out")\nprint("err", file=sys.stderr)\nsys.exit(3)\n')
        r = await tools.t_run_command(self.sb, py_command(p))
        # Windows PowerShell -Command reports 1 for a failed native command, bash its code.
        self.assertIn('exit=1' if sys.platform == 'win32' else 'exit=3', r.text.split('\n', 1)[0])
        self.assertIn('out', r.text)
        self.assertIn('--- stderr ---\nerr', r.text)
        self.assertTrue(r.error)
        self.assertNotIn('still running', r.text)


# --------------------------------------------------------------------------- malformed calls
class MalformedCallTests(unittest.IsolatedAsyncioTestCase):
    """Class: malformed tool calls. Recorded: 8 write_file calls cut off inside the content
    string at max_output_tokens=32000 (24-32k reasoning tokens + arguments, completion
    exactly 32000), 1 run_command cut by a broken stream, 1 prepare_edit_plan with a literal
    control character, hallucinated `bash`/`bash_placeholder`, calls missing a required
    argument, a stale `job_wait`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'a.txt').write_text('one\ntwo\n', encoding='utf-8')

    async def test_cut_off_call_at_the_output_limit(self):
        s = make_session(self.root, reasoning_effort='high', max_output_tokens=32000)
        raw = '{"path": "big.py", "content": "import numpy as np\\n' + 'x = 1\\n' * 50
        provider = ScriptedProvider(
            [{'type': 'tool_calls', 'tool_calls': [call('w1', 'write_file', raw=raw)]},
             {'type': 'done', 'finish_reason': 'length'}],
            text_turn('ok'))
        with patch.object(agent.llm, 'stream_chat', provider):
            await run(s)
        result = next(x for x in s.conv.segments if x.kind == K_TOOL_RESULT)
        self.assertIn('the call was cut off', result.body)
        self.assertIn('max_output_tokens=32000', result.body)
        self.assertIn('Nothing was executed', result.body)
        self.assertFalse((self.root / 'big.py').exists())
        self.assertEqual([r['reasoning_effort'] for r in provider.requests], ['high', 'medium'])
        self.assertEqual(provider.problems, [])  # the broken arguments were re-encoded on the wire

    def test_truncated_without_length_finish_and_plain_invalid_json(self):
        name, args, err = tool_batch.parse_call(call('x', 'run_command', raw='{"command": "echo hi'))
        self.assertIn('end abruptly', err.text)
        name, args, err = tool_batch.parse_call(call('x', 'run_command', raw='{"command": echo}'))
        self.assertIn('not valid JSON', err.text)
        name, args, err = tool_batch.parse_call(call('x', 'run_command', raw='[1, 2]'))
        self.assertIn('JSON object', err.text)

    def test_literal_control_characters_are_accepted(self):
        raw = '{"command": "printf \'a\tb\'\nexit 0"}'
        with self.assertRaises(ValueError):
            json.loads(raw)
        name, args, err = tool_batch.parse_call(call('x', 'run_command', raw=raw))
        self.assertIsNone(err)
        self.assertEqual(args['command'], "printf 'a\tb'\nexit 0")

    async def test_hallucinated_and_disabled_tools_name_the_real_one(self):
        s = make_session(self.root)
        r = await s._exec_tool('bash', {'command': 'ls'})
        self.assertTrue(r.error)
        self.assertIn('Did you mean run_command', r.text)
        r = await s._exec_tool('bash_placeholder', {})
        self.assertIn('run_command', r.text)
        s.cfg.enabled_tools = [n for n in s.cfg.enabled_tools if n != 'job_wait']
        r = await s._exec_tool('job_wait', {'job_id': 'x'})
        self.assertIn('not enabled for this chat', r.text)
        r = await s._exec_tool('reed_file', {'path': 'a.txt'})
        self.assertIn('read_file', r.text)

    async def test_missing_and_unknown_arguments_list_the_parameters(self):
        s = make_session(self.root)
        r = await s._exec_tool('write_file', {'path': 'b.txt'})
        self.assertTrue(r.error)
        self.assertIn("missing required argument 'content'", r.text)
        self.assertIn('content (string, required)', r.text)
        self.assertFalse((self.root / 'b.txt').exists())
        r = await s._exec_tool('read_file', {'path': 'a.txt', 'lines': 5})
        self.assertIn("unknown argument 'lines'", r.text)
        self.assertIn('offset (integer)', r.text)
        r = await s._exec_tool('pin', {'segment_ids': ['x'], 'forever': True})
        self.assertIn("unknown argument 'forever'", r.text)  # used to raise out of the batch

    async def test_string_scalars_are_converted_not_truthy(self):
        s = make_session(self.root)
        (self.root / 'c.txt').write_text('a a a', encoding='utf-8')
        r = await s._exec_tool('edit_file', {'path': 'c.txt', 'old_string': 'a', 'new_string': 'b',
                                              'replace_all': 'false'})
        # "false" was truthy before: it replaced every occurrence instead of failing as ambiguous.
        self.assertIn('occurs 3 times', r.text)
        self.assertEqual((self.root / 'c.txt').read_text(encoding='utf-8'), 'a a a')
        r = await s._exec_tool('read_file', {'path': 'a.txt', 'offset': '2', 'limit': '1'})
        self.assertIn('two', r.text)
        self.assertNotIn('one', r.text.split('\n', 1)[1])
        r = await s._exec_tool('edit_file', {'path': 'c.txt', 'old_string': 'a', 'new_string': 'b',
                                              'replace_all': 'sometimes'})
        self.assertIn("must be true or false", r.text)


# --------------------------------------------------------------------------- repeated failing calls
class RepeatedFailureTests(unittest.IsolatedAsyncioTestCase):
    """Class: the same failing call repeated unchanged. Recorded only once (an interrupted
    run_command issued twice), so this is a guard, not a measured loop: the second identical
    failure says that repeating it will not help, unless something changed in between."""

    async def test_second_identical_failure_is_flagged_and_edits_reset_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.txt').write_text('alpha', encoding='utf-8')
            s = make_session(root)
            miss = {'path': 'a.txt', 'old_string': 'beta', 'new_string': 'gamma'}
            provider = ScriptedProvider(
                tool_turn(call('e1', 'edit_file', miss)),
                tool_turn(call('e2', 'edit_file', miss)),
                tool_turn(call('w1', 'write_file', {'path': 'a.txt', 'content': 'alpha beta'})),
                tool_turn(call('e3', 'edit_file', {'path': 'a.txt', 'old_string': 'delta', 'new_string': 'x'})),
                tool_turn(call('e4', 'edit_file', {'path': 'a.txt', 'old_string': 'delta', 'new_string': 'x'})),
                text_turn('ok'))
            with patch.object(agent.llm, 'stream_chat', provider):
                await run(s)
        results = {x.tool_call_id: x.body for x in s.conv.segments if x.kind == K_TOOL_RESULT}
        self.assertNotIn('[harness] This exact call', results['e1'])
        self.assertIn('[harness] This exact call already failed the same way 1 time(s)', results['e2'])
        self.assertNotIn('[harness]', results['e3'])  # different call
        self.assertIn('[harness]', results['e4'])
        # A successful write in between resets the memory for the first call's shape too.
        s2 = make_session()
        r = tools.ToolResult('Error: x', error=True)
        s2._note_repeated_failure('read_file', call('a', 'read_file', {'path': 'z'}), r)
        s2._note_repeated_failure('write_file', call('b', 'write_file', {}), tools.ToolResult('ok', mutates=True))
        r2 = tools.ToolResult('Error: x', error=True)
        s2._note_repeated_failure('read_file', call('c', 'read_file', {'path': 'z'}), r2)
        self.assertNotIn('[harness]', r2.text)


# --------------------------------------------------------------------------- schema portability
class SchemaPortabilityTests(unittest.TestCase):
    """No recorded rejection, but the tool schemas go to OpenAI, Anthropic and (through
    OpenRouter) Gemini-style endpoints: keep them inside the subset all of them accept.
    `document_open.select` used `oneOf`, which Gemini function declarations do not support."""

    def walk(self, node, path, out):
        if isinstance(node, dict):
            for key in ('oneOf', 'allOf', 'not', '$ref', 'patternProperties', 'if'):
                if key in node:
                    out.append(f'{path}: {key}')
            if node.get('type') == 'array' and 'items' not in node:
                out.append(f'{path}: array without items')
            if node.get('type') == 'object' and 'properties' in node:
                out += [f'{path}: required {r!r} not in properties'
                        for r in node.get('required') or [] if r not in node['properties']]
            for k, v in node.items():
                self.walk(v, f'{path}.{k}', out)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                self.walk(v, f'{path}[{i}]', out)

    def test_every_tool_schema_is_portable(self):
        import re
        problems = []
        for name, spec in tools.REGISTRY.items():
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
                problems.append(f'{name}: name')
            if (spec.parameters or {}).get('type') != 'object':
                problems.append(f'{name}: parameters must be an object schema')
            self.walk(spec.parameters, name, problems)
        self.assertEqual(problems, [])


def fake_anthropic_module():
    import types
    mod = types.ModuleType('anthropic')

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, message, status_code=500):
            super().__init__(message)
            self.status_code, self.message = status_code, message

    class BadRequestError(APIStatusError):
        pass

    class APIConnectionError(APIError):
        pass
    mod.APIError, mod.APIStatusError, mod.BadRequestError, mod.APIConnectionError = (
        APIError, APIStatusError, BadRequestError, APIConnectionError)
    return mod


class _NS:
    """Attribute bag with the SDK's model_dump()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class _FakeSdkStream:
    def __init__(self, events, exc=None):
        self.events, self.exc = events, exc

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e
        if self.exc is not None:
            raise self.exc

    async def close(self):
        pass


MSG_START = _NS(type='message_start', message=_NS(id='msg', model='claude-opus-5-5', usage=_NS(input_tokens=5)))
TEXT_START = _NS(type='content_block_start', index=0, content_block=_NS(type='text', text=''))
MSG_STOP = _NS(type='message_delta', delta=_NS(stop_reason='end_turn', stop_details=None), usage=None)


def text_delta(text):
    return _NS(type='content_block_delta', index=0, delta=_NS(type='text_delta', text=text))


class AnthropicDirectRetryTests(unittest.IsolatedAsyncioTestCase):
    """The direct Anthropic route: the SDK retries only the initial request; an
    `overloaded_error` mid-stream before any content block is now sent again, after a
    content block nothing is replayed. (The anthropic SDK is not needed: a stand-in
    module provides its exception types.)"""

    async def run_direct(self, *streams):
        from ctxagent import anthropic_api as aa
        fake = fake_anthropic_module()
        queue = [_FakeSdkStream(ev, fake.APIStatusError(err, 529) if err else None) for ev, err in streams]
        calls = []

        async def create(**kw):
            calls.append(kw)
            return queue.pop(0)

        async def no_sleep(_):
            return None
        client = _NS(beta=_NS(messages=_NS(create=create)))
        with patch.dict(sys.modules, {'anthropic': fake}), patch.object(aa, '_get_client', return_value=client), \
                patch.object(llm, '_sleep', no_sleep):
            events = [e async for e in aa.stream_chat(model='claude-opus-5-5',
                                                       messages=[{'role': 'user', 'content': 'hi'}])]
        return events, calls

    async def test_overloaded_before_content_is_retried(self):
        events, calls = await self.run_direct(
            ([MSG_START], 'Overloaded'), ([MSG_START, TEXT_START, text_delta('ok'), MSG_STOP], None))
        self.assertEqual(len(calls), 2)
        self.assertEqual(events[-1], {'type': 'done', 'finish_reason': 'stop'})
        self.assertTrue(any(e['type'] == 'notice' for e in events))

    async def test_failure_after_content_is_not_replayed(self):
        events, calls = await self.run_direct(([MSG_START, TEXT_START, text_delta('par')], 'Overloaded'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertTrue(events[-1]['interrupted'])


if __name__ == "__main__":
    unittest.main(verbosity=1)
