"""Offline LaTeX analysis and edit-plan tests. Compiler smoke is separate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ctxagent import edit_plans, latex_workflow, tools


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sb = tools.Sandbox([str(self.root)])

    def test_source_graph_labels_refs_cites_comments_and_cycle(self):
        main = self.root / 'main.tex'
        main.write_text(r'\input{chapter}\label{duplicate}\ref{missing}\cite{known,unknown}' + '\n% \\input{absent}\n' + r'\bibliography{refs}', encoding='utf-8')
        (self.root / 'chapter.tex').write_text(r'\input{main}\label{duplicate}', encoding='utf-8')
        (self.root / 'refs.bib').write_text('@article{known, title={Verified}}', encoding='utf-8')
        result = latex_workflow.inspect(self.sb, 'main.tex')
        self.assertEqual(len(result['files']), 2)
        self.assertIn('duplicate', result['duplicate_labels'])
        self.assertIn('missing', result['unresolved_references'])
        self.assertEqual(set(result['unresolved_citations']), {'unknown'})
        self.assertFalse(result['missing_inputs'])

    def test_outside_inputs_not_followed(self):
        (self.root / 'main.tex').write_text(r'\input{../outside}\input{\dynamic}', encoding='utf-8')
        result = latex_workflow.inspect(self.sb, 'main.tex')
        self.assertEqual(len(result['blocked_inputs']), 1)
        self.assertEqual(len(result['dynamic_inputs']), 1)

    def test_log_diagnostics(self):
        p = self.root / 'main.log'
        p.write_text('main.tex:12: Undefined control sequence.\n! Emergency stop.\n'
                     'LaTeX Warning: Reference `missing` undefined.\n'
                     'Overfull \\hbox (12.0pt too wide) in paragraph at lines 3--4\n'
                     'Fatal error occurred, no output PDF file produced!\n', encoding='utf-8')
        result = latex_workflow.diagnostics(p)
        self.assertTrue(result['fatal'])
        self.assertFalse(result['output_written'])
        self.assertEqual(result['entries'][0]['line'], 12)
        self.assertTrue(any(e['severity'] == 'layout' for e in result['entries']))

    def test_plan_preview_apply_rollback_and_crlf_bom(self):
        p = self.root / 'file.tex'
        raw = b'\xef\xbb\xbfalpha\r\nbeta\r\n'
        p.write_bytes(raw)
        plan = edit_plans.prepare(self.sb, 'test-owner', [{'path': str(p), 'edits': [
            {'old_string': 'alpha\nbeta', 'new_string': 'gamma\ndelta'}]}])
        self.assertEqual(p.read_bytes(), raw)
        self.assertIn('-alpha', plan['diff'])
        result = edit_plans.apply(self.sb, 'test-owner', plan['plan_id'])
        self.assertEqual(result['state'], 'applied')
        self.assertEqual(p.read_bytes(), b'\xef\xbb\xbfgamma\r\ndelta\r\n')
        edit_plans.apply(self.sb, 'test-owner', plan['plan_id'], rollback=True)
        self.assertEqual(p.read_bytes(), raw)

    def test_plan_all_preconditions_before_any_write(self):
        a, b = self.root / 'a.py', self.root / 'b.py'
        a.write_text('first'); b.write_text('second')
        plan = edit_plans.prepare(self.sb, 'test-owner', [
            {'path': str(a), 'edits': [{'old_string': 'first', 'new_string': 'changed'}]},
            {'path': str(b), 'edits': [{'old_string': 'second', 'new_string': 'changed'}]}])
        b.write_text('external change')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            edit_plans.apply(self.sb, 'test-owner', plan['plan_id'])
        self.assertEqual(a.read_text(), 'first')
        with self.assertRaises(PermissionError):
            edit_plans.apply(self.sb, 'other-owner', plan['plan_id'])

    def test_rollback_refuses_external_change(self):
        p = self.root / 'a.txt'; p.write_text('before')
        plan = edit_plans.prepare(self.sb, 'test-owner', [{'path': str(p), 'edits': [
            {'old_string': 'before', 'new_string': 'after'}]}])
        edit_plans.apply(self.sb, 'test-owner', plan['plan_id'])
        p.write_text('user change')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            edit_plans.apply(self.sb, 'test-owner', plan['plan_id'], rollback=True)
        self.assertEqual(p.read_text(), 'user change')

    def test_plan_rejects_ambiguous_empty_and_stale_expected(self):
        p = self.root / 'a.txt'; p.write_text('same same')
        for old in ('same', ''):
            with self.assertRaises(ValueError):
                edit_plans.prepare(self.sb, 'test', [{'path': str(p), 'edits': [{'old_string': old, 'new_string': 'x'}]}])
        with self.assertRaisesRegex(ValueError, 'Expected hash'):
            edit_plans.prepare(self.sb, 'test', [{'path': str(p), 'expected_sha256': 'bad',
                'edits': [{'old_string': 'same same', 'new_string': 'x'}]}])

    def test_partial_failure_has_recoverable_journal_without_overwriting(self):
        a, b = self.root / 'a.txt', self.root / 'b.txt'
        a.write_text('before-a'); b.write_text('before-b')
        plan = edit_plans.prepare(self.sb, 'test-owner', [
            {'path': str(a), 'edits': [{'old_string': 'before-a', 'new_string': 'after-a'}]},
            {'path': str(b), 'edits': [{'old_string': 'before-b', 'new_string': 'after-b'}]}])
        original = edit_plans.atomic_bytes
        def injected(path, raw, **kwargs):
            if path == b:
                raise OSError('simulated interrupted write')
            original(path, raw, **kwargs)
        with patch.object(edit_plans, 'atomic_bytes', injected):
            with self.assertRaises(OSError):
                edit_plans.apply(self.sb, 'test-owner', plan['plan_id'])
        self.assertEqual(a.read_text(), 'after-a')
        self.assertEqual(b.read_text(), 'before-b')
        import json
        manifest = json.loads((edit_plans.directory(plan['plan_id']) / 'plan.json').read_text())
        self.assertEqual(manifest['applied'], [0])
        self.assertEqual(manifest['pending_index'], 1)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            edit_plans.apply(self.sb, 'test-owner', plan['plan_id'], rollback=True)

    def test_build_flags_and_no_rc(self):
        p = self.root / 'main.tex'; p.write_text('test')
        with patch.object(latex_workflow.shutil, 'which', return_value='latexmk'), \
             patch.object(latex_workflow.jobs, 'start', return_value={'id': 'fake'}) as start:
            result = latex_workflow.build(self.sb, 'main.tex', 'test')
        self.assertIn('-norc', result['argv'])
        self.assertIn('-latexoption=-no-shell-escape', result['argv'])
        self.assertEqual(start.call_args.args[2], 'test')


if __name__ == '__main__':
    unittest.main(verbosity=2)
