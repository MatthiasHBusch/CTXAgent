"""Bounded read-only concurrency; every other tool is an ordering barrier.

A multi-call model response already costs one model request. Concurrency saves
wall time, NOT additional cache billing by itself. Batch planning saves requests.
Results commit in model order; shell tools and context mutations never overlap.
"""
import asyncio
import json
import time

from .tools import RUN_CANCEL, ToolResult

MAX_PARALLEL_READS = 4
READ_ONLY = frozenset({'read_file', 'read_document', 'focus_file', 'list_dir',
                       'glob_search', 'grep_search', 'summarize_folder'})


def is_parallel_read(name, args):
    if name == 'document_call':
        from .document_formats import READ_ONLY as scoped_reads
        return args.get('operation') in scoped_reads
    return name in READ_ONLY or (name == 'summarize_file' and
        str(args.get('mode', 'static')).strip().lower() in ('static', 'outline', 'structure'))


def parse_call(tc, cut_off_at=None):
    """(name, args, error result or None). `cut_off_at` is the max_output_tokens setting when
    the response that carried this call stopped at the output limit."""
    fn = tc.get('function') or {}
    name = fn.get('name') or '?'
    raw = fn.get('arguments')
    if raw is None or raw == '':
        raw = '{}'
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    try:
        # strict=False: literal newlines/tabs inside strings (recorded from Claude models)
        # are unambiguous, so accept them instead of failing the call.
        args = json.loads(raw, strict=False)
        if not isinstance(args, dict):
            raise ValueError('arguments must be a JSON object')
        return name, args, None
    except (ValueError, TypeError) as exc:
        return name, {}, ToolResult(bad_arguments_text(name, raw, exc, cut_off_at), error=True)


def bad_arguments_text(name, raw, exc, cut_off_at=None):
    """Recorded: write_file calls whose JSON ends inside the content string because the
    response hit max_output_tokens (most of the budget went to reasoning). The bare JSON
    error ('Unterminated string starting at ...') did not say that, nor what to do."""
    head = f'Bad arguments for {name}: '
    if cut_off_at:
        return (head + f'the call was cut off. The response reached the output limit '
                + (f'(max_output_tokens={cut_off_at}) ' if cut_off_at is not True else "(the model's maximum) ")
                + 'while these arguments were being written; they end '
                f'after {len(raw)} characters ({exc}). Nothing was executed. Most of the output budget '
                'went to private reasoning, so the next request runs with lower reasoning effort. Write '
                'long content in parts (write_file with the first part, then edit_file to add the rest) '
                'or generate it with a short script.')
    if not raw.rstrip().endswith('}'):
        return (head + f'the arguments end abruptly after {len(raw)} characters ({exc}); the call was '
                'probably cut off. Nothing was executed. Send the complete call again; split very long '
                'content across several calls.')
    return head + f'not valid JSON ({exc}). Nothing was executed. Send the arguments as one JSON object.'


async def execute_batch(session, calls, finish_reason=None):
    """Yield (phase, call, name, args, result, ms); owns and cleans up tasks."""
    async def execute(name, args, invalid):
        start = time.perf_counter()
        if session.cancel.is_set():
            result = ToolResult('Tool not executed: user cancelled this batch.', error=True)
        else:
            if isinstance(getattr(session, 'cancel', None), asyncio.Event):
                RUN_CANCEL.set(session.cancel)  # this task's context only
            result = invalid or await session._exec_tool(name, args)
        return result, int((time.perf_counter() - start) * 1000)

    capped = str(finish_reason or '').lower() in ('length', 'max_tokens', 'max_output_tokens')
    cut = (getattr(getattr(session, 'cfg', None), 'max_output_tokens', None) or True) if capped else None

    def parse(tc):
        return parse_call(tc, cut)

    i = 0
    tasks = []
    try:
        while i < len(calls):
            group = [calls[i]]
            name, args, invalid = parse(calls[i])
            if is_parallel_read(name, args) and invalid is None:
                while i + len(group) < len(calls) and len(group) < MAX_PARALLEL_READS:
                    n, a, bad = parse(calls[i + len(group)])
                    if bad is not None or not is_parallel_read(n, a):
                        break
                    group.append(calls[i + len(group)])
            parsed = [parse(tc) for tc in group]
            for tc, (name, args, invalid) in zip(group, parsed):
                yield 'start', tc, name, args, None, 0
            tasks = [asyncio.create_task(execute(*parts)) for parts in parsed]
            for tc, (name, args, _), task in zip(group, parsed, tasks):
                result, ms = await task
                yield 'result', tc, name, args, result, ms
            tasks = []
            i += len(group)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
