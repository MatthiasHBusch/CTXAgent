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
            'Keep a file as a live view outside the history, re-read from disk before each request. select picks sections (names or Lx-y, as in focus_file); without it the whole text. Opening a path again replaces its view. A view over max_tokens or over the budget shared with browser tabs is refused.',
            _obj({'path': _s(), 'select': {'anyOf': [_s(), {'type': 'array', 'items': _s()}]},
                  'max_tokens': _i('Per-view budget, 1..20000 (default 8000).')}, ['path']),
            document_open, group='documents', needs_ctx=True),
        ToolSpec('document_close',
            'Remove a live view. summary (up to 1000 tokens) stays in the conversation as a note.',
            _obj({'document_id': _s(), 'summary': _s()}, ['document_id']),
            document_close, group='documents', needs_ctx=True),
        ToolSpec('document_list', 'Open documents: ids, views, versions, errors, token costs.',
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
            "Prepare exact-string edits for up to 30 open documents as one plan and return its diff; nothing is written until document_apply_edits. Edits within a file run in order. expected_version is the document's current workspace version. BOM and CRLF are preserved.",
            _obj({'changes': array(_obj({**fields, 'edits': array(_obj({
                'old_string': _s(), 'new_string': _s(), 'replace_all': {'type': 'boolean'}},
                ['old_string', 'new_string']), 100)},
                ['document_id', 'expected_version', 'edits']), 30)}, ['changes']),
            document_prepare_edits, group='documents', needs_ctx=True),
        ToolSpec('document_json_update',
            'Prepare a plan replacing existing nodes of an open .json document, addressed by RFC 6901 pointers (/key/0, ~1 = slash, ~0 = tilde, empty = root). value_json is JSON text. No append or delete; formatting may be normalised. Returns the diff; document_apply_edits writes it.',
            _obj({**fields, 'updates': array(_obj({'pointer': _s(), 'value_json': _s()},
                  ['pointer', 'value_json']), 100)}, ['document_id', 'expected_version', 'updates']),
            document_json_update, group='documents', needs_ctx=True),
        ToolSpec('document_table_update',
            'Prepare a plan of cell edits for an open CSV or TSV with unique headers. row is the zero-based data row (header excluded), column the exact header, value a string. No inserted rows or columns; quoting may be normalised. Returns the diff; document_apply_edits writes it.',
            _obj({**fields, 'updates': array(_obj({'row': _i('Zero-based data row, header excluded.'),
                  'column': _s(), 'value': _s()}, ['row', 'column', 'value']), 100)},
                  ['document_id', 'expected_version', 'updates']),
            document_table_update, group='documents', needs_ctx=True),
        ToolSpec('document_apply_edits',
            'Write a prepared plan while its documents are still open and unchanged on disk (hashes are rechecked). Files are replaced one by one with backups, not as one transaction; rollback=true restores the backups. Returns the new versions.',
            _obj({'plan_id': _s(), 'rollback': {'type': 'boolean'}}, ['plan_id']),
            document_apply_edits, group='documents', dangerous=True, needs_ctx=True),
    ]
    for spec in specs:
        REGISTRY[spec.name] = spec
