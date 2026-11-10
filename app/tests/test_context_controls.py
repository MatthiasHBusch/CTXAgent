"""Regressions for the 2026-10-15 context-control changes, without model calls.

* grep_search never returns a silent false negative (large files, globs, limits).
* the `pin` tool: pin / timed expiry / unpin / restore of a shortened result.
* an empty model reply does not end a turn silently.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from ctxagent import tools
from ctxagent.compaction import CompactionEngine, curate
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_USER
from tests.test_compaction import fake_complete


def batch(conv, n=1, lines=400):
    ids = [f'c{len(conv)}-{i}' for i in range(n)]
    conv.new('assistant', K_TOOL_CALL, '', tool_calls=[
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}} for cid in ids])
    return [conv.new('tool', K_TOOL_RESULT, '\n'.join(f'{j}\tline {j} of {cid}' for j in range(lines)),
                     tool_name='read_file', tool_call_id=cid) for cid in ids]


class GrepTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'src' / 'deep').mkdir(parents=True)
        (self.root / 'big.md').write_text(('filler line\n' * 600_000) + 'needle at the end\n', encoding='utf-8')
        (self.root / 'src' / 'deep' / 'a.cpp').write_text('int x;\n// needle here\nint y;\n', encoding='utf-8')
        (self.root / 'src' / 'b.gd').write_text('var needle = 1\n', encoding='utf-8')
        (self.root / 'bin.dat').write_bytes(b'needle\0\0\0')
        (self.root / '.hidden').mkdir()
        (self.root / '.hidden' / 'h.txt').write_text('needle hidden\n', encoding='utf-8')
        self.sb = tools.Sandbox([str(self.root)])

    def tearDown(self):
        self.tmp.cleanup()

    async def grep(self, **kw):
        return (await tools.t_grep_search(self.sb, path=str(self.root), **kw)).text

    async def test_large_file_is_searched(self):
        self.assertGreater((self.root / 'big.md').stat().st_size, 4_000_000)
        t = await self.grep(pattern='needle at the end')
        self.assertIn('big.md', t)
        self.assertIn('1 hit(s)', t)

    async def test_glob_double_star_matches_zero_dirs_and_deep(self):
        t = await self.grep(pattern='needle', glob='src/**/*.cpp')
        self.assertIn('a.cpp', t)
        t2 = await self.grep(pattern='needle', glob='**/*.gd')
        self.assertIn('b.gd', t2)

    async def test_reports_skips_and_limits(self):
        t = await self.grep(pattern='needle')
        self.assertIn('binary skipped', t)
        self.assertIn('.hidden', t)          # reported as not searched
        self.assertNotIn('h.txt', t)
        t2 = await self.grep(pattern='filler', max_results=3)
        self.assertIn('STOPPED', t2)
        self.assertIn('+ hit(s)', t2)
        t3 = await self.grep(pattern='needle', glob='*.nomatch')
        self.assertIn('glob matched none', t3)

    async def test_explicit_hidden_path_and_modes(self):
        r = await tools.t_grep_search(self.sb, pattern='needle', path=str(self.root / '.hidden'))
        self.assertIn('h.txt', r.text)
        t = await self.grep(pattern='NEEDLE', case_insensitive=True, mode='count')
        self.assertRegex(t, r'\d+ hit\(s\)')
        t = await self.grep(pattern='needle here', context=1, glob='*.cpp')
        self.assertIn('int x;', t)
        self.assertIn('int y;', t)
        t = await self.grep(pattern='var needle = (', literal=True)
        self.assertIn('0 hit(s)', t)


class LineEndingTests(unittest.IsolatedAsyncioTestCase):
    """edit_file / write_file keep the file's own line ending (Windows used to turn LF into CRLF)."""

    async def test_lf_and_crlf_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sb = tools.Sandbox([d])
            lf, crlf = root / 'lf.txt', root / 'crlf.txt'
            lf.write_bytes(b'a\nb\nc\n')
            crlf.write_bytes(b'a\r\nb\r\nc\r\n')
            await tools.t_edit_file(sb, str(lf), 'b\nc', 'B\nC\nD')
            await tools.t_edit_file(sb, str(crlf), 'b\nc', 'B\nC\nD')   # LF in the patch, CRLF file
            self.assertEqual(lf.read_bytes(), b'a\nB\nC\nD\n')
            self.assertEqual(crlf.read_bytes(), b'a\r\nB\r\nC\r\nD\r\n')
            await tools.t_write_file(sb, str(lf), 'x\ny\n')
            self.assertEqual(lf.read_bytes(), b'x\ny\n')
            await tools.t_write_file(sb, str(crlf), 'x\ny\n')
            self.assertEqual(crlf.read_bytes(), b'x\r\ny\r\n')
            new = root / 'new.txt'
            await tools.t_write_file(sb, str(new), 'p\nq\n')
            self.assertEqual(new.read_bytes(), b'p\nq\n')


class PinToolTests(unittest.IsolatedAsyncioTestCase):
    def session(self, conv):
        from tests.test_productivity import make_session
        return make_session(conv)

    async def test_pin_expiry_and_restore(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys', pinned=True)
        target = batch(c, 1)[0]
        full = target.body
        target.rewrite('stub', 'masked', 'test')
        s = self.session(c)
        r = s._pin_tool(segment_ids=[target.id], batches=2)
        self.assertFalse(r.error, r.text)
        self.assertTrue(target.pinned)
        self.assertEqual(target.body, full)          # restored to the full original
        self.assertIn('restored', r.text)
        cfg = CompactionConfig(rewrite_interval_batches=100)
        eng = CompactionEngine('t', cfg, fake_complete)
        for _ in range(2):
            batch(c, 1)
            await eng.run(c, 2_000_000)
            self.assertTrue(target.pinned)
        batch(c, 1)
        r2 = await eng.run(c, 2_000_000)
        self.assertFalse(target.pinned)
        self.assertEqual(r2.get('pins_expired'), 1)

    async def test_pin_last_list_unpin(self):
        c = Conversation()
        prev = batch(c, 2)
        batch(c, 1)                                   # the batch issuing the pin call
        s = self.session(c)
        r = s._pin_tool(segment_ids=['last'])
        self.assertTrue(all(x.pinned and x.pin_until_batch is None for x in prev), r.text)
        listing = s._pin_tool(list_pins=True).text
        self.assertIn('2 pinned result(s)', listing)
        s._pin_tool(segment_ids=[prev[0].id], unpin=True)
        self.assertFalse(prev[0].pinned)
        bad = s._pin_tool(segment_ids=['nope'])
        self.assertTrue(bad.error)


class ManualCompactionTests(unittest.IsolatedAsyncioTestCase):
    SUMMARY = ('Handoff: task A (the torch freeze) is finished. Fix in chunk_manager.cpp, guard in '
               'regenerate_chunk_sync_body; tests 1710 E9 green; nothing open. ' * 3)

    def build(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys', pinned=True)
        c.new('user', K_USER, 'task A')
        old = [batch(c, 2) for _ in range(5)]
        c.new('user', K_USER, 'task B now')
        new = batch(c, 1)
        running = batch(c, 1)          # the batch carrying the compact_context call
        return c, old, new, running

    def test_compacts_before_last_user_and_keeps_rest(self):
        from ctxagent.compaction import manual
        c, old, new, running = self.build()
        keep = old[2][0]
        keep.pinned, keep.pin_scope = True, 'manual'      # a manual pin stays verbatim
        before = c.total_tokens()
        rep = manual.compact(c, 't', summary=self.SUMMARY)
        self.assertGreater(rep['saved_tokens'], 0)
        self.assertLess(c.total_tokens(), before)
        kinds = [s.kind for s in c.segments]
        self.assertEqual(kinds[0], K_SYSTEM)
        self.assertEqual(kinds[1], 'summary')
        ids = {s.id for s in c.segments}
        self.assertIn(keep.id, ids)                       # kept verbatim, with its group
        self.assertIn(old[2][1].id, ids)
        self.assertNotIn(old[0][0].id, ids)               # archived ...
        self.assertIsNotNone(c.archived_by_id(old[0][0].id))   # ... but recallable
        self.assertTrue(all(s.id in ids for s in new + running))
        self.assertIn(old[0][0].id, c.segments[1].body)   # the index names archived results

    def test_guards_and_until(self):
        from ctxagent.compaction import manual
        c, old, new, running = self.build()
        n = len(c.segments)
        with self.assertRaises(ValueError):
            manual.compact(c, 't', summary='   ')     # empty: nothing would remain (no minimum length any more)
        with self.assertRaises(ValueError):
            manual.compact(c, 't', summary=self.SUMMARY, until=999)
        self.assertEqual(len(c.segments), n)
        rep2 = manual.compact(c, 't', summary=self.SUMMARY, until=1)   # up to the newest item: all but the running batch
        self.assertIsNotNone(rep2['cache_breakeven_requests'])
        self.assertTrue(all(s.id in {x.id for x in c.segments} for s in running))
        self.assertNotIn(new[0].id, {x.id for x in c.segments})
        self.assertGreater(rep2['saved_tokens'], 0)


class AgentCuratedReviewTests(unittest.IsolatedAsyncioTestCase):
    def cfg(self, **kw):
        base = dict(rewrite_policy='agent', rewrite_interval_batches=3, deep_enabled=False)
        base.update(kw)
        return CompactionConfig(**base)

    def build(self, n=6):
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys', pinned=True)
        c.new('user', K_USER, 'go')
        return c

    async def test_due_cycle_hints(self):
        c = self.build()
        eng = CompactionEngine('t', self.cfg(), fake_complete)
        reports = []
        for _ in range(3):
            batch(c, 1)
            reports.append(await eng.run(c, 2_000_000))
        self.assertTrue(all(s.state == 'original' for s in c.segments if s.kind == K_TOOL_RESULT))
        self.assertIn('review_hint', reports[-1])
        newest = [s for s in c.segments if s.kind == K_TOOL_RESULT][-1]
        self.assertTrue(newest.body.endswith(curate.HINT))
        # not repeated on the very next batch
        batch(c, 1)
        r = await eng.run(c, 2_000_000)
        self.assertNotIn('review_hint', r)
        self.assertIn('agent review due', r['schedule']['reason'])

    async def test_ignored_hints_change_nothing(self):
        c = self.build()
        eng = CompactionEngine('t', self.cfg(), fake_complete)
        hints = 0
        for _ in range(9):
            batch(c, 1)
            hints += 'review_hint' in await eng.run(c, 2_000_000)
        self.assertEqual(hints, 3)
        self.assertTrue(all(s.state == 'original' for s in c.segments if s.kind == K_TOOL_RESULT))

    async def test_deep_pressure_is_still_automatic(self):
        c = self.build()
        eng = CompactionEngine('t', self.cfg(rewrite_interval_batches=100, deep_enabled=True, deep_trigger_ratio=.75,
                                             deep_min_gain_tokens=100, deep_cooldown_turns=0), fake_complete)
        for _ in range(4):
            batch(c, 1)
        r = await eng.run(c, int(c.total_tokens() / 0.8))
        self.assertTrue(r['deep'])
        self.assertTrue(any(s.kind == 'summary' for s in c.segments))



if __name__ == '__main__':
    unittest.main()
