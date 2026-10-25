"""Bounded read-only concurrency; every other tool is an ordering barrier.

A multi-call model response already costs one model request. Concurrency saves
wall time, NOT additional cache billing by itself. Batch planning saves requests.
Results commit in model order; shell tools and context mutations never overlap.
"""
import asyncio
import json
import time

from .tools import ToolResult

MAX_PARALLEL_READS = 4
READ_ONLY = frozenset({'read_file', 'read_document', 'focus_file', 'list_dir',
                       'glob_search', 'grep_search', 'summarize_folder'})


def is_parallel_read(name, args):
    if name == 'document_call':
        from .document_formats import READ_ONLY as scoped_reads
        return args.get('operation') in scoped_reads
    return name in READ_ONLY or (name == 'summarize_file' and
        str(args.get('mode', 'static')).strip().lower() in ('static', 'outline', 'structure'))


def parse_call(tc):
    fn = tc.get('function') or {}
    name = fn.get('name') or '?'
    try:
        raw = fn.get('arguments') or '{}'
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError('arguments must be an object')
        return name, args, None
    except (ValueError, TypeError) as exc:
        return name, {}, ToolResult(f'Bad arguments for {name}: {exc}', error=True)


async def execute_batch(session, calls):
    """Yield (phase, call, name, args, result, ms); owns and cleans up tasks."""
    async def execute(name, args, invalid):
        start = time.perf_counter()
        if session.cancel.is_set():
            result = ToolResult('Tool not executed: user cancelled this batch.', error=True)
        else:
            result = invalid or await session._exec_tool(name, args)
        return result, int((time.perf_counter() - start) * 1000)

    i = 0
    tasks = []
    try:
        while i < len(calls):
            group = [calls[i]]
            name, args, invalid = parse_call(calls[i])
            if is_parallel_read(name, args) and invalid is None:
                while i + len(group) < len(calls) and len(group) < MAX_PARALLEL_READS:
                    n, a, bad = parse_call(calls[i + len(group)])
                    if bad is not None or not is_parallel_read(n, a):
                        break
                    group.append(calls[i + len(group)])
            parsed = [parse_call(tc) for tc in group]
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
