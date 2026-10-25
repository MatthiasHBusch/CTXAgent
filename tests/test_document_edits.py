"""Offline adapter/scope/refresh regressions; never touch the live chat store."""
import asyncio
import codecs
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_DATA = tempfile.TemporaryDirectory(prefix='ctxagent-document-edits-')
os.environ.setdefault('CTXAGENT_DATA', _DATA.name)

from ctxagent import document_edits as edits, edit_plans
from ctxagent.agent import AgentSession
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import Conversation
from ctxagent.tools import Sandbox
from ctxagent.workspace import DocumentWorkspace, WorkspaceError


class DocumentEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sb = Sandbox([self.tmp.name])
        self.w = DocumentWorkspace()

    def open(self, name, text, select=None):
        p = self.root / name
        p.write_bytes(text.encode('utf-8') if isinstance(text, str) else text)
        doc = self.w.open(self.sb, str(p), select)
        return p, doc, self.version(doc)

    def version(self, doc):
        return next(v['version'] for v in self.w.listing()['views'] if v['id'] == doc)

    def change(self, doc, version, old, new):
        return {'document_id': doc, 'expected_version': version,
                'edits': [{'old_string': old, 'new_string': new}]}

    def test_source_diagnostic_is_data_but_missing_dependency_is_error(self):
        p, doc, version = self.open('source.py', 'ERROR = "is not installed — run `pip install x`"\n')
        self.assertIn('ERROR =', self.w.messages()[0]['content'])
        p.write_text('placeholder', encoding='utf-8')
        pdf = self.root / 'source.pdf'
        pdf.write_bytes(b'placeholder')
        with patch('ctxagent.workspace.readers.extract_text',
                   return_value='[`fitz` is not installed — run `pip install pymupdf` to read this format]'):
            with self.assertRaises(WorkspaceError):
                self.w.open(self.sb, str(pdf))

    def test_multiedit_review_apply_refresh_and_rollback_preserve_bytes(self):
        original = codecs.BOM_UTF8 + b'first\r\nsecond\r\n'
        p, doc, version = self.open('a.txt', original)
        q, other, v2 = self.open('b.md', 'old\n')
        change = self.change(doc, version, 'first', 'intermediate')
        change['edits'].append({'old_string': 'intermediate', 'new_string': 'final'})
        plan = edits.prepare(self.sb, 'owner', self.w, [change, self.change(other, v2, 'old', 'new')])
        self.assertIn('+final', plan['diff'])
        self.assertEqual(p.read_bytes(), original)
        self.assertEqual(q.read_text(), 'old\n')
        result = edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        self.assertEqual(p.read_bytes(), codecs.BOM_UTF8 + b'final\r\nsecond\r\n')
        self.assertEqual(q.read_text(), 'new\n')
        self.assertNotEqual(self.version(doc), version)
        self.assertEqual(len(result['views']), 2)
        self.assertIn('final', self.w.messages()[0]['content'])
        edits.apply(self.sb, 'owner', self.w, plan['plan_id'], rollback=True)
        self.assertEqual(p.read_bytes(), original)
        self.assertEqual(q.read_text(), 'old\n')

    def test_closed_target_invalidates_plan_even_if_other_document_open(self):
        p, doc, version = self.open('a.txt', 'old')
        self.open('other.txt', 'other')
        plan = edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'old', 'new')])
        self.w.close(doc)
        with self.assertRaisesRegex(ValueError, 'open and ready'):
            edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        self.assertEqual(p.read_text(), 'old')
        # Reopening changes its id; an old scope must not accidentally bind it.
        self.w.open(self.sb, str(p))
        with self.assertRaises(ValueError):
            edits.apply(self.sb, 'owner', self.w, plan['plan_id'])

    def test_version_checks_before_prepare_and_stale_apply(self):
        p, doc, version = self.open('a.txt', 'old')
        for bad in (None, '', '0' * 64):
            with self.assertRaises(ValueError):
                edits.prepare(self.sb, 'owner', self.w, [self.change(doc, bad, 'old', 'new')])
        p.write_text('external', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'File changed'):
            edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'old', 'new')])
        self.w.refresh(self.sb)
        with self.assertRaisesRegex(ValueError, 'version mismatch'):
            edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'old', 'new')])
        plan = edits.prepare(self.sb, 'owner', self.w,
                             [self.change(doc, self.version(doc), 'external', 'new')])
        p.write_text('drift', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Stale file'):
            edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        self.assertEqual(p.read_text(), 'drift')

    def test_owner_and_general_plan_cannot_bypass_scope(self):
        p, doc, version = self.open('a.txt', 'old')
        plan = edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'old', 'new')])
        with self.assertRaises(PermissionError):
            edits.apply(self.sb, 'other', self.w, plan['plan_id'])
        general = edit_plans.prepare(self.sb, 'owner', [{'path': str(p), 'edits': [
            {'old_string': 'old', 'new_string': 'new'}]}])
        with self.assertRaisesRegex(ValueError, 'Not a document-scoped plan'):
            edits.apply(self.sb, 'owner', self.w, general['plan_id'])

    def test_selected_source_can_prepare_reviewed_local_patch(self):
        p, doc, version = self.open('a.py', 'def a():\n    return 1\n\ndef b():\n    return 2\n', 'a')
        plan = edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'return 1', 'return 3')])
        edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        self.assertIn('return 3', self.w.messages()[0]['content'])
        self.assertNotIn('return 2', self.w.messages()[0]['content'])

    def test_json_pointer_crlf_bom_and_root_updates(self):
        p, doc, version = self.open('a.json', codecs.BOM_UTF8 +
            b'{\r\n  "a/b": {"~x": [1, true]},\r\n  "name": "old"\r\n}\r\n')
        plan = edits.json_update(self.sb, 'owner', self.w, doc, version,
                                [{'pointer': '/a~1b/~0x/0', 'value_json': '9'},
                                 {'pointer': '/name', 'value_json': '"updated"'}])
        edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        raw = p.read_bytes()
        self.assertTrue(raw.startswith(codecs.BOM_UTF8))
        self.assertNotIn(b'\n', raw.replace(b'\r\n', b''))
        data = json.loads(raw.decode('utf-8-sig'))
        self.assertEqual(data, {'a/b': {'~x': [9, True]}, 'name': 'updated'})
        plan = edits.json_update(self.sb, 'owner', self.w, doc, self.version(doc),
                                 [{'pointer': '', 'value_json': '{"root": 4}'}])
        edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
        self.assertEqual(json.loads(p.read_bytes().decode('utf-8-sig')), {'root': 4})

    def test_json_rejects_invalid_inputs_without_writes(self):
        p, doc, version = self.open('a.json', '{"x": [1]}')
        bad_updates = [[], [{'pointer': '/missing', 'value_json': '2'}],
            [{'pointer': '/x/01', 'value_json': '2'}], [{'pointer': '/x/-', 'value_json': '2'}],
            [{'pointer': '/bad~2', 'value_json': '2'}], [{'pointer': '/x', 'value_json': 'NaN'}],
            [{'pointer': '/x', 'value_json': '{"k": 1, "k": 2}'}],
            [{'pointer': '/x', 'value_json': '1.234567890123456789'}],
            [{'pointer': '/x', 'value_json': '1e309'}],
            [{'pointer': '/x', 'value_json': '1e-324'}],
            [{'pointer': '/x', 'value_json': '1'}, {'pointer': '/x', 'value_json': '2'}]]
        for updates in bad_updates:
            with self.subTest(updates=updates), self.assertRaises((ValueError, TypeError)):
                edits.json_update(self.sb, 'owner', self.w, doc, version, updates)
            self.assertEqual(p.read_text(), '{"x": [1]}')
        p.write_text('{"x": 1, "x": 2}', encoding='utf-8')
        self.w.refresh(self.sb)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            edits.json_update(self.sb, 'owner', self.w, doc, self.version(doc),
                              [{'pointer': '/x', 'value_json': '3'}])

    def test_csv_and_tsv_cell_updates_keep_strings_and_rectangular_data(self):
        for suffix, separator, newline in [('csv', ',', '\r\n'), ('tsv', '\t', '\n')]:
            with self.subTest(suffix=suffix):
                text = f'id{separator}note{newline}001{separator}old{newline}002{separator}keep{newline}'
                p, doc, version = self.open(f'a.{suffix}', codecs.BOM_UTF8 + text.encode('utf-8'))
                value = 'a,quoted "value"\nsecond line'
                plan = edits.table_update(self.sb, 'owner', self.w, doc, version,
                                          [{'row': 0, 'column': 'note', 'value': value}])
                self.assertEqual(p.read_bytes(), codecs.BOM_UTF8 + text.encode('utf-8'))
                edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
                rows = list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'), newline=''),
                                       delimiter=separator))
                self.assertEqual(rows, [['id', 'note'], ['001', value], ['002', 'keep']])
                self.assertTrue(p.read_bytes().startswith(codecs.BOM_UTF8))

    def test_table_rejects_ambiguous_headers_ragged_rows_and_bad_indices(self):
        p, doc, version = self.open('a.csv', 'id,note\n001,old\n')
        for update in [{'row': -1, 'column': 'note', 'value': 'x'},
                       {'row': 1, 'column': 'note', 'value': 'x'},
                       {'row': True, 'column': 'note', 'value': 'x'},
                       {'row': 0, 'column': 'missing', 'value': 'x'},
                       {'row': 0, 'column': 'note', 'value': 3}]:
            with self.assertRaises(ValueError):
                edits.table_update(self.sb, 'owner', self.w, doc, version, [update])
        update = {'row': 0, 'column': 'note', 'value': 'x'}
        with self.assertRaisesRegex(ValueError, 'Duplicate cell'):
            edits.table_update(self.sb, 'owner', self.w, doc, version, [update, update])
        for text in ['id,id\n1,2\n', 'id,note\n1\n', 'id,\n1,2\n']:
            p.write_text(text, encoding='utf-8')
            self.w.refresh(self.sb)
            with self.assertRaises(ValueError):
                edits.table_update(self.sb, 'owner', self.w, doc, self.version(doc), [update])

    def test_windows_atomic_retry_rechecks_source_and_cleans_temp(self):
        import os
        p = self.root / 'locked.txt'
        p.write_bytes(b'before')
        original = os.replace
        failure = PermissionError('simulated Windows sharing conflict')
        failure.winerror = 32
        attempts = 0
        def transient(source, target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise failure
            return original(source, target)
        with patch('ctxagent.edit_plans.os.replace', side_effect=transient) as replace, \
             patch('ctxagent.edit_plans.time.sleep'):
            edit_plans.atomic_bytes(p, b'after', expected_sha256=edit_plans.sha(b'before'))
            self.assertEqual(replace.call_count, 2)
        self.assertEqual(p.read_bytes(), b'after')
        self.assertEqual(list(self.root.glob('*.ctxedit')), [])
        p.write_bytes(b'before')

        def concurrent_change(source, target):
            p.write_bytes(b'external')
            raise failure
        with patch('ctxagent.edit_plans.os.replace', side_effect=concurrent_change), \
             patch('ctxagent.edit_plans.time.sleep'), \
             self.assertRaisesRegex(ValueError, 'changed during atomic'):
            edit_plans.atomic_bytes(p, b'after', expected_sha256=edit_plans.sha(b'before'))
        self.assertEqual(p.read_bytes(), b'external')
        self.assertEqual(list(self.root.glob('*.ctxedit')), [])

    def test_non_windows_permission_error_is_not_retried(self):
        p = self.root / 'locked.txt'
        p.write_bytes(b'before')
        with patch('ctxagent.edit_plans.os.replace', side_effect=PermissionError('ACL')) as replace, \
             patch('ctxagent.edit_plans.time.sleep') as sleep, self.assertRaises(PermissionError):
            edit_plans.atomic_bytes(p, b'after', expected_sha256=edit_plans.sha(b'before'))
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(p.read_bytes(), b'before')
        self.assertEqual(list(self.root.glob('*.ctxedit')), [])

    def test_table_embedded_cr_lf_values_survive_subsequent_edits(self):
        for newline in ('\n', '\r\n'):
            p, doc, version = self.open('multiline.csv', f'id,note{newline}001,old{newline}')
            value = 'CR\rLF\nCRLF\r\nUnicode \u2028 text'
            plan = edits.table_update(self.sb, 'owner', self.w, doc, version,
                                      [{'row': 0, 'column': 'note', 'value': value}])
            edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
            plan = edits.table_update(self.sb, 'owner', self.w, doc, self.version(doc),
                                      [{'row': 0, 'column': 'id', 'value': '002'}])
            edits.apply(self.sb, 'owner', self.w, plan['plan_id'])
            data = list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8'), newline='')))
            self.assertEqual(data, [['id', 'note'], ['002', value]])

    def test_wrong_adapter_and_non_utf8_sources_are_refused(self):
        p, doc, version = self.open('a.txt', 'old')
        with self.assertRaisesRegex(ValueError, 'format'):
            edits.json_update(self.sb, 'owner', self.w, doc, version,
                              [{'pointer': '', 'value_json': '3'}])
        p, doc, version = self.open('binary.txt', b'old\x00binary')
        with self.assertRaisesRegex(ValueError, 'Binary'):
            edits.prepare(self.sb, 'owner', self.w, [self.change(doc, version, 'old', 'new')])


class DynamicDocumentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_follows_open_close_readiness_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            session = AgentSession('schema-test', SessionConfig(workdirs=[td]), Conversation())
            w = session.conv.workspace
            self.assertFalse(set(edits.TOOLS) & set(session.tool_names()))
            json_path = Path(td) / 'data.json'
            json_path.write_text('{"n": 1}', encoding='utf-8')
            doc = w.open(session.sandbox(), str(json_path))
            names = {t['function']['name'] for t in session.tool_schema()}
            self.assertIn('document_call', names)
            self.assertFalse(set(edits.TOOLS) & names)
            from ctxagent import document_catalog
            operations = document_catalog.operation_names(session)
            self.assertIn('document_json_update', operations)
            self.assertIn('document_prepare_edits', operations)
            self.assertNotIn('document_table_update', operations)
            session.cfg.enabled_tools.remove('document_json_update')
            self.assertNotIn('document_json_update', session.tool_names())
            session.cfg.enabled_tools.append('document_json_update')
            json_path.unlink()
            w.refresh(session.sandbox())
            self.assertFalse(set(edits.TOOLS) & set(session.tool_names()))
            json_path.write_text('{"n": 1}', encoding='utf-8')
            w.refresh(session.sandbox())
            self.assertIn('document_json_update', session.tool_names())
            w.close(doc)
            result = await session._exec_tool('document_json_update',
                         {'document_id': doc, 'expected_version': '0' * 64,
                          'updates': [{'pointer': '/n', 'value_json': '2'}]})
            self.assertTrue(result.error)
            self.assertIn('not enabled', result.text)

    async def test_real_tool_roundtrip_returns_current_views_without_duplicate_contents(self):
        with tempfile.TemporaryDirectory() as td:
            session = AgentSession('tool-test', SessionConfig(workdirs=[td],
                                   compaction=CompactionConfig(enabled=False)), Conversation())
            p = Path(td) / 'data.json'
            p.write_text('{"n": 1}', encoding='utf-8')
            opened = await session._exec_tool('document_open', {'path': str(p)})
            self.assertFalse(opened.error)
            view = session.conv.workspace.listing()['views'][0]
            prepared = await session._exec_tool('document_json_update',
                         {'document_id': view['id'], 'expected_version': view['version'],
                          'updates': [{'pointer': '/n', 'value_json': '2'}]})
            self.assertFalse(prepared.error, prepared.text)
            plan = json.loads(prepared.text)
            self.assertEqual(json.loads(p.read_text()), {'n': 1})
            result = await session._exec_tool('document_apply_edits', {'plan_id': plan['plan_id']})
            self.assertFalse(result.error, result.text)
            self.assertEqual(json.loads(p.read_text()), {'n': 2})
            self.assertIn('"n":2', session.conv.workspace.messages()[0]['content'])
            self.assertNotIn('"n":2', result.text)
            receipt_view = json.loads(result.text)['views'][0]
            self.assertNotIn('body', receipt_view)
            self.assertLessEqual(set(receipt_view), {'id', 'path', 'select', 'max_tokens', 'version',
                                                   'status', 'error', 'representation', 'tokens'})
            self.assertNotEqual(receipt_view['version'], view['version'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
