"""Live document lifecycle and opt-in, document-scoped editing tools.

Receipts contain metadata rather than repeating the open document. Specialized
edit adapters are registered here; AgentSession exposes them only while a
compatible, ready document is open. They prepare reviewable plans, not blind
writes. Applying a plan rechecks both the live documents and source hashes.
"""
import asyncio
import json

from .tools import ToolResult, ToolSpec, _p


def _obj(props, required=None):
    return _p(props, required or [])


def _s():
    return {'type': 'string'}


def _i(description):
    return {'type': 'integer', 'description': description}


def register():
    from .tools import REGISTRY

    async def document_open(sb, path, select=None, max_tokens=8000, ctx=None):
        session = ctx['session']
        from .workspace import DocumentWorkspace, WorkspaceError
        from .browser_workspace import shared_budget
        candidate = DocumentWorkspace.from_dict(session.conv.workspace.to_dict())
        await asyncio.to_thread(candidate.refresh, sb)
        doc_id = await asyncio.to_thread(candidate.open, sb, path, select, max_tokens)
        if candidate.token_count() + session.conv.browser.token_count() > shared_budget(session.cfg):
            raise WorkspaceError('Shared document/browser live-context budget exceeded; close views first')
        session.conv.workspace = candidate
        view = next(v for v in session.conv.workspace.listing()['views'] if v['id'] == doc_id)
        return ToolResult('Opened live document. Content is supplied once in the workspace on the next request.\n'
                          + json.dumps(view, ensure_ascii=False))

    async def document_close(sb, document_id, summary='', ctx=None):
        from .workspace import WorkspaceError
        from .tokens import count
        from .segments import K_USER
        if count(summary) > 1000:
            raise WorkspaceError('Close summary must be at most 1000 tokens.')
        session = ctx['session']
        path = session.conv.workspace.close(document_id)
        if summary.strip():
            session.conv.new('user', K_USER, f'Document handoff for {path} (close note, not file evidence):\n{summary}')
        return ToolResult(f'Closed {path}; live content removed. Historical reads are not erased. '
                          + ('Handoff note retained.' if summary.strip() else 'No handoff note supplied.'))

    async def document_list(sb, ctx=None):
        return ToolResult(json.dumps(ctx['session'].conv.workspace.listing(), ensure_ascii=False))

    specs = [
        ToolSpec('document_open',
            'Keep a current document in a live workspace, refreshed from disk before each model request. '
            'Omit select for full source/extracted text; select names or Lx-y for sections. '
            'Opening the same path replaces its view. Content appears once outside frozen history; '
            'oversized views are refused, never silently truncated. Use focus_file first to discover selectors.',
            _obj({'path': _s(), 'select': {'oneOf': [_s(), {'type': 'array', 'items': _s()}]},
                  'max_tokens': _i('Per-view budget, 1..20000 (default 8000).')}, ['path']),
            document_open, group='documents', needs_ctx=True),
        ToolSpec('document_close',
            'Remove a live document view by id. Optional summary retains a concise handoff note '
            '(max 1000 tokens); existing historical reads remain recoverable.',
            _obj({'document_id': _s(), 'summary': _s()}, ['document_id']),
            document_close, group='documents', needs_ctx=True),
        ToolSpec('document_list', 'List open document ids, views, versions, errors and token costs (no content).',
                 _obj({}), document_list, group='documents', needs_ctx=True),
    ]
    from . import document_edits

    def edit_result(value):
        return ToolResult(json.dumps(value, ensure_ascii=False, indent=2))

    async def document_prepare_edits(sb, changes, ctx=None):
        session = ctx['session']
        return edit_result(await asyncio.to_thread(document_edits.prepare, sb, session.chat_id,
                                                  session.conv.workspace, changes))

    async def document_json_update(sb, document_id, expected_version, updates, ctx=None):
        session = ctx['session']
        return edit_result(await asyncio.to_thread(document_edits.json_update, sb, session.chat_id,
                            session.conv.workspace, document_id, expected_version, updates))

    async def document_table_update(sb, document_id, expected_version, updates, ctx=None):
        session = ctx['session']
        return edit_result(await asyncio.to_thread(document_edits.table_update, sb, session.chat_id,
                            session.conv.workspace, document_id, expected_version, updates))

    async def document_apply_edits(sb, plan_id, rollback=False, ctx=None):
        session = ctx['session']
        return edit_result(await asyncio.to_thread(document_edits.apply, sb, session.chat_id,
                                                  session.conv.workspace, plan_id, rollback))

    array = lambda items, maximum: {'type': 'array', 'items': items, 'minItems': 1, 'maxItems': maximum}
    fields = {'document_id': _s(), 'expected_version': _s()}
    specs += [
        ToolSpec('document_prepare_edits',
            'Prepare ONE reviewed multi-edit plan for 1..30 OPEN, ready UTF-8 documents. '
            'Use document_id and current workspace version; edits within each file run in order. '
            'Returns complete diff/backups without changing sources. Selected views may omit affected '
            'regions: review the diff. Refuses stale files; preserves BOM/CRLF. Then document_apply_edits.',
            _obj({'changes': array(_obj({**fields, 'edits': array(_obj({
                'old_string': _s(), 'new_string': _s(), 'replace_all': {'type': 'boolean'}},
                ['old_string', 'new_string']), 100)},
                ['document_id', 'expected_version', 'edits']), 30)}, ['changes']),
            document_prepare_edits, group='documents', needs_ctx=True),
        ToolSpec('document_json_update',
            'Prepare a reviewed JSON edit plan for an OPEN, ready .json document. '
            'Replace existing nodes with RFC 6901 pointers (/key/0, ~1 slash, ~0 tilde; empty=root). '
            'value_json is JSON-encoded text, not a literal cell string. Updates run in order; '
            'no append/delete, duplicate keys/pointers or NaN. Serialization may normalize formatting; '
            'review the full diff before document_apply_edits.',
            _obj({**fields, 'updates': array(_obj({'pointer': _s(), 'value_json': _s()},
                  ['pointer', 'value_json']), 100)}, ['document_id', 'expected_version', 'updates']),
            document_json_update, group='documents', needs_ctx=True),
        ToolSpec('document_table_update',
            'Prepare a reviewed cell-edit plan for an OPEN, ready comma CSV or tab TSV with unique headers. '
            'row is zero-based DATA row, excluding header; column is exact header; value is always a string. '
            'No type inference, row/column insertion, duplicate edits or ragged tables. Quoting may normalize; '
            'review the full diff before document_apply_edits.',
            _obj({**fields, 'updates': array(_obj({'row': _i('Zero-based data row, header excluded.'),
                  'column': _s(), 'value': _s()}, ['row', 'column', 'value']), 100)},
                  ['document_id', 'expected_version', 'updates']),
            document_table_update, group='documents', needs_ctx=True),
        ToolSpec('document_apply_edits',
            'Apply a reviewed document-scoped plan only while ALL its target documents remain open and ready. '
            'Rechecks source hashes, writes serially with per-file atomic replacement/backups/journal, '
            'NOT a global transaction. rollback=true restores backups if hashes still match. '
            'Refreshes live views and returns new versions, not full contents.',
            _obj({'plan_id': _s(), 'rollback': {'type': 'boolean'}}, ['plan_id']),
            document_apply_edits, group='documents', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        REGISTRY[spec.name] = spec
