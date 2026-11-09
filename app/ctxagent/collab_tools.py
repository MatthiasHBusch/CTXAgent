"""Tools for working together with the user: the shared scratchpad (scratchpad.py)."""
from __future__ import annotations

import asyncio
import json

SCRATCHPAD_TOOLS = ["scratchpad_read", "scratchpad_write"]


def register(registry, ToolSpec, ToolResult, ToolError, params):
    from . import scratchpad

    def chat(ctx):
        if not ctx or not ctx.get('session'):
            raise ToolError('This tool requires a chat session')
        return ctx['session'].chat_id

    async def scratchpad_read(sb, ctx=None):
        state = await asyncio.to_thread(scratchpad.get, chat(ctx))
        head = {k: state[k] for k in ('version', 'updated_by')}
        return ToolResult(text=json.dumps(head, ensure_ascii=False) + '\n<scratchpad>\n'
                          + (state['text'] or '(empty)') + '\n</scratchpad>')

    async def scratchpad_write(sb, text=None, append=None, edits=None, ctx=None):
        try:
            state = await asyncio.to_thread(scratchpad.agent_write, chat(ctx), text=text, append=append, edits=edits)
        except scratchpad.ScratchpadError as exc:
            return ToolResult(text=f'scratchpad_write: {exc}. Nothing changed.', error=True)
        return ToolResult(text=json.dumps({'version': state['version'], 'chars': len(state['text']),
                                           'note': 'The user sees the change in the scratchpad panel.'}))

    registry['scratchpad_read'] = ToolSpec(
        'scratchpad_read',
        'Read the shared scratchpad: a short text the user and you both edit, shown permanently beside '
        'the chat. The user sends it to you with Ctrl+Enter (a message starting with [scratchpad] '
        'carries its text); read it yourself when you need its current version.',
        params({}, []), scratchpad_read, 'collaboration', needs_ctx=True)
    registry['scratchpad_write'] = ToolSpec(
        'scratchpad_write',
        'Write in the shared scratchpad the user sees beside the chat. Pass exactly one of: text '
        '(replaces everything), append (adds a line at the end), edits (old_string -> new_string '
        'replacements; each old_string must occur exactly once). Max 20000 characters. The user may '
        'be typing in it too: prefer append or small edits over replacing their text.',
        params({'text': {'type': 'string'}, 'append': {'type': 'string'},
                'edits': {'type': 'array', 'minItems': 1, 'maxItems': 50, 'items': {
                    'type': 'object', 'properties': {'old_string': {'type': 'string'},
                                                     'new_string': {'type': 'string'}},
                    'required': ['old_string', 'new_string']}}}, []),
        scratchpad_write, 'collaboration', needs_ctx=True)
