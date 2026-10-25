"""Document-scoped editing: checked live sources -> reviewable multi-file plans.

Preparation never changes source files. All writes use the existing edit-plan
journal/backups and stay serial. These adapters are deliberately limited to
UTF-8 text, JSON and rectangular CSV/TSV, plus the conservative native adapters
in document_formats. Format schemas are delivered through document_catalog.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import re

from . import edit_plans, readers
from .jobs import atomic_json

from .document_formats import BY_SUFFIX, NAMES

TOOLS = ('document_prepare_edits', 'document_json_update',
         'document_table_update', 'document_apply_edits') + NAMES
NON_TEXT = {'pdf', 'docx', 'pptx', 'excel', 'parquet', 'odt', 'epub'}


def tools_for_path(path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() in BY_SUFFIX:
        return list(BY_SUFFIX[path.suffix.lower()])
    kind = readers.detect_kind(path)
    if kind in NON_TEXT:
        return []
    names = ['document_prepare_edits', 'document_apply_edits']
    if kind == 'json':
        names.append('document_json_update')
    if kind in ('csv', 'tsv'):
        names.append('document_table_update')
    return names


def available(name, workspace) -> bool:
    """No I/O here: refresh before the request determines schema availability."""
    if name not in TOOLS:
        return True
    return any(v.get('status') == 'ready' and name in tools_for_path(v['path'])
               for v in workspace.views)


def _checked(sb, workspace, document_id, expected_version=None, tool=None, *, binary=False):
    view = next((v for v in workspace.views if v['id'] == document_id), None)
    if view is None or view.get('status') != 'ready':
        raise ValueError('Document must be open and ready; reopen or refine its live view.')
    path = sb.resolve(view['path'], must_exist=True)
    if tool and tool not in tools_for_path(path):
        raise ValueError('This document format does not support the requested adapter.')
    if tool != 'document_apply_edits' and (
            not isinstance(expected_version, str) or not re.fullmatch(r'[a-f0-9]{64}', expected_version)):
        raise ValueError('Provide the current 64-character workspace version.')
    if expected_version is not None and view['version'] != expected_version:
        raise ValueError('Expected live version mismatch; use the current workspace version.')
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Edit source exceeds 8 MiB.')
    raw = path.read_bytes()
    if expected_version is not None and edit_plans.sha(raw) != expected_version:
        raise ValueError('File changed since the live view; wait for refresh before preparing edits.')
    if binary:
        return view, path, raw, None
    text = raw.decode('utf-8-sig', errors='strict')
    if '\x00' in text:
        raise ValueError('Binary sources are not supported.')
    return view, path, raw, text


def _plan(sb, owner, changes, scopes, note='', *, verbatim=False):
    result = edit_plans.prepare(sb, owner, changes, verbatim=verbatim)
    path = edit_plans.directory(result['plan_id']) / 'plan.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['documents'] = scopes
    atomic_json(path, manifest)
    result['next'] = 'Review the complete diff, then dispatch document_apply_edits via document_call with {plan_id}. No source files changed yet.'
    if note:
        result['note'] = note
    return result


def prepare(sb, owner, workspace, changes):
    if not isinstance(changes, list) or not 1 <= len(changes) <= 30:
        raise ValueError('Provide 1..30 open-document changes.')
    files, scopes = [], []
    for change in changes:
        view, path, raw, text = _checked(sb, workspace, change['document_id'],
                                       change['expected_version'], 'document_prepare_edits')
        files.append({'path': str(path), 'expected_sha256': edit_plans.sha(raw),
                      'edits': change['edits']})
        scopes.append({'id': view['id'], 'path': str(path)})
    return _plan(sb, owner, files, scopes,
                 'Edits in one file run in order. Selected views may omit affected regions; review the diff. '
                 'UTF-8 only; BOM and uniform CRLF are preserved. Application is serial, not a global transaction.')


def _reject_constant(value):
    raise ValueError(f'Non-standard JSON constant: {value}')


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'Duplicate JSON key: {key}')
        out[key] = value
    return out


def _safe_float(raw):
    value = float(raw)
    if Decimal(str(value)) != Decimal(raw):
        raise ValueError('JSON number would lose precision; use an exact text patch instead.')
    return value


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
                      parse_float=_safe_float)


def _pointer_parts(pointer):
    if pointer == '':
        return []
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Use an RFC 6901 JSON Pointer (empty string selects the root).')
    parts = pointer[1:].split('/')
    if any(re.search(r'~(?![01])', p) for p in parts):
        raise ValueError('Invalid JSON Pointer escape.')
    return [p.replace('~1', '/').replace('~0', '~') for p in parts]


def _key(node, part):
    if isinstance(node, dict):
        if part not in node:
            raise ValueError(f'JSON field does not exist: {part}')
        return part
    if isinstance(node, list) and re.fullmatch(r'0|[1-9][0-9]*', part):
        index = int(part)
        if index < len(node):
            return index
    raise ValueError(f'JSON Pointer cannot address: {part}')


def _newline(text):
    if '\r' in text.replace('\r\n', ''):
        raise ValueError('Bare-CR line endings are not supported by structured adapters.')
    if '\r\n' in text and '\n' in text.replace('\r\n', ''):
        raise ValueError('Mixed line endings are not supported by structured adapters.')
    return '\r\n' if '\r\n' in text else '\n'


def _structured_plan(sb, owner, view, path, raw, text, output, note):
    if not text or output == text:
        raise ValueError('Update would not change the document.')
    return _plan(sb, owner,
                 [{'path': str(path), 'expected_sha256': edit_plans.sha(raw),
                   'edits': [{'old_string': text, 'new_string': output}]}],
                 [{'id': view['id'], 'path': str(path)}], note, verbatim=True)


def json_update(sb, owner, workspace, document_id, expected_version, updates):
    view, path, raw, text = _checked(sb, workspace, document_id, expected_version, 'document_json_update')
    if not isinstance(updates, list) or not 1 <= len(updates) <= 100:
        raise ValueError('Provide 1..100 JSON updates.')
    data = _json(text)
    seen = set()
    for update in updates:
        pointer = update['pointer']
        if pointer in seen:
            raise ValueError('Duplicate JSON Pointer; combine edits to the same field.')
        seen.add(pointer)
        parts = _pointer_parts(pointer)
        value = _json(update['value_json'])
        if not parts:
            data = value
            continue
        parent = data
        for part in parts[:-1]:
            parent = parent[_key(parent, part)]
        parent[_key(parent, parts[-1])] = value
    newline = _newline(text)
    indent_match = re.search(r'(?m)^([ \t]+)\S', text)
    indent = indent_match.group(1) if indent_match else (2 if '\n' in text else None)
    output = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=indent,
                        separators=(',', ':') if indent is None else None)
    output = output.replace('\n', newline)
    if text.endswith('\n'):
        output += newline
    return _structured_plan(sb, owner, view, path, raw, text, output,
                            'Existing JSON nodes only (no append/delete). Updates run in order. '
                            'Serialization may normalize whitespace/escapes; review the full diff. BOM/newlines preserved.')


def _table_records(text, delimiter):
    # csv.reader can consume several physical lines for one quoted record.
    # Measure record endings, not embedded cell newlines.
    stream = io.StringIO(text, newline='')
    rows, endings = [], set()
    for row in csv.reader(stream, delimiter=delimiter, strict=True):
        rows.append(row)
        end = stream.tell()
        tail = text[max(0, end - 2):end]
        if tail.endswith('\r\n'):
            endings.add('\r\n')
        elif tail.endswith('\n'):
            endings.add('\n')
        elif tail.endswith('\r'):
            raise ValueError('Bare-CR record endings are not supported.')
    if len(endings) > 1:
        raise ValueError('Mixed record endings are not supported.')
    return rows, next(iter(endings), '\n')


def table_update(sb, owner, workspace, document_id, expected_version, updates):
    view, path, raw, text = _checked(sb, workspace, document_id, expected_version, 'document_table_update')
    if not isinstance(updates, list) or not 1 <= len(updates) <= 100:
        raise ValueError('Provide 1..100 cell updates.')
    delimiter = '\t' if readers.detect_kind(path) == 'tsv' else ','
    rows, newline = _table_records(text, delimiter)
    if not rows or not rows[0] or len(set(rows[0])) != len(rows[0]) or any(not h for h in rows[0]):
        raise ValueError('Table needs a nonempty, unique header row.')
    header = rows[0]
    if any(len(row) != len(header) for row in rows):
        raise ValueError('Table must be rectangular; blank/ragged rows are not supported.')
    seen = set()
    for update in updates:
        row, column, value = update['row'], update['column'], update['value']
        if type(row) is not int or not 0 <= row < len(rows) - 1:
            raise ValueError('Row must be a zero-based DATA row (header excluded), within the table.')
        if column not in header or not isinstance(value, str):
            raise ValueError('Use an exact column header and a string cell value.')
        key = (row, column)
        if key in seen:
            raise ValueError('Duplicate cell update; combine edits to the same cell.')
        seen.add(key)
        rows[row + 1][header.index(column)] = value
    stream = io.StringIO(newline='')
    # The LF dialect does not necessarily quote a bare CR in a cell.
    quoting = csv.QUOTE_ALL if newline == '\n' and any('\r' in c for r in rows for c in r) else csv.QUOTE_MINIMAL
    writer = csv.writer(stream, delimiter=delimiter, lineterminator=newline, quoting=quoting)
    writer.writerows(rows)
    output = stream.getvalue()
    if not text.endswith('\n'):
        output = output[:-len(newline)]
    if _table_records(output, delimiter)[0] != rows:
        raise ValueError('CSV roundtrip validation failed; no source changed.')
    return _structured_plan(sb, owner, view, path, raw, text, output,
                            'Comma CSV or tab TSV with unique headers only. All cells remain strings; no type inference. '
                            'Serialization may normalize quoting; review the full diff. BOM/newlines preserved.')


def apply(sb, owner, workspace, plan_id, rollback=False):
    manifest = json.loads((edit_plans.directory(plan_id) / 'plan.json').read_text(encoding='utf-8'))
    if manifest['owner'] != owner:
        raise PermissionError('Edit plan belongs to another chat.')
    if not manifest.get('documents'):
        raise ValueError('Not a document-scoped plan; use apply_edit_plan for general plans.')
    for scope in manifest['documents']:
        view, path, raw, text = _checked(sb, workspace, scope['id'], tool='document_apply_edits', binary=True)
        if str(path) != scope['path']:
            raise ValueError('Document path changed; refusing to apply the plan.')
    result = edit_plans.apply(sb, owner, plan_id, rollback)
    # The request loop also refreshes before the next request. Refresh here so
    # a later call in this same batch can use the returned current versions.
    workspace.refresh(sb)
    result['views'] = [v for v in workspace.listing()['views']
                       if v['id'] in {s['id'] for s in manifest['documents']}]
    return result
