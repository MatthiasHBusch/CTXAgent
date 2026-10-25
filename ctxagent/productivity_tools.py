"""Stage-one tool adapters. Implementations are independently testable services."""
from __future__ import annotations

import asyncio
import json

from . import artifacts, jobs


def register(registry, ToolSpec, ToolResult, ToolError, params):
    def result(data, **kwargs):
        return ToolResult(text=json.dumps(data, ensure_ascii=False, indent=2), **kwargs)

    def owner(ctx):
        if not ctx or not ctx.get('session'):
            raise ToolError('This tool requires a chat session')
        return ctx['session'].chat_id

    async def view_image(sb, path: str, max_side: int = 1600):
        p = sb.resolve(path, must_exist=True)
        images = await asyncio.to_thread(artifacts.view, p, max_side)
        return result({'source': str(p), 'images': images,
                       'note': 'Native images follow this tool batch. Requires a vision-capable model.'},
                      images=images)

    async def render_pdf(sb, path: str, pages: list[int] | None = None, dpi: int = 120):
        p = sb.resolve(path, must_exist=True)
        images = await asyncio.to_thread(artifacts.render, p, pages, dpi)
        return result({'source': str(p), 'images': images,
                       'note': 'Native page images follow this tool batch. Text extraction alone cannot verify layout.'},
                      images=images)

    async def start_job(sb, command: str, cwd: str = '.', timeout: int = 3600, ctx=None):
        wd = sb.resolve(cwd, must_exist=True)
        data = await asyncio.to_thread(jobs.start, command, wd, owner(ctx), timeout)
        return result(data)

    async def job_status(sb, job_id: str, ctx=None):
        return result(await asyncio.to_thread(jobs.status, job_id, owner(ctx)))

    async def job_wait(sb, job_id: str, seconds: int = 120, tail_bytes: int = 2000, ctx=None):
        import time
        who = owner(ctx)
        seconds = max(1, min(600, int(seconds)))
        t0 = time.monotonic()
        first = await asyncio.to_thread(jobs.status, job_id, who)
        st = first
        terminal = set(jobs.TERMINAL)
        while st.get('state') not in terminal and time.monotonic() - t0 < seconds:
            await asyncio.sleep(min(2.0, max(0.2, seconds - (time.monotonic() - t0))))
            st = await asyncio.to_thread(jobs.status, job_id, who)
        out = {'state': st.get('state'), 'exit_code': st.get('exit_code'),
               'waited_s': round(time.monotonic() - t0, 1),
               'finished': st.get('state') in terminal}
        tail_bytes = max(0, min(20000, int(tail_bytes)))
        if tail_bytes:
            for stream in ('stdout', 'stderr'):
                size = int((st.get('log_bytes') or {}).get(stream, 0) or 0)
                if size:
                    chunk = await asyncio.to_thread(jobs.read_log, job_id, who, stream,
                                                    max(0, size - tail_bytes), tail_bytes)
                    out[f'{stream}_tail'] = chunk.get('text', '')
        return result(out)

    async def list_jobs(sb, ctx=None):
        return result(await asyncio.to_thread(jobs.list_owned, owner(ctx)))

    async def read_job_log(sb, job_id: str, stream: str = 'stdout', cursor: int = 0,
                           max_bytes: int = 16000, ctx=None):
        return result(await asyncio.to_thread(jobs.read_log, job_id, owner(ctx), stream, cursor, max_bytes))

    async def cancel_job(sb, job_id: str, ctx=None):
        return result(await asyncio.to_thread(jobs.cancel, job_id, owner(ctx)))

    definitions = [
        ToolSpec('view_image', 'Inspect an image visually: sends native image pixels to the model and a UI preview. '
                 'Normalizes orientation and first frame. Requires vision-capable model. PNG/JPEG/WebP etc.; not SVG.',
                 params({'path': {'type': 'string'}, 'max_side': {'type': 'integer', 'minimum': 256,
                         'maximum': 2400, 'description': 'Longest side limit; default 1600.'}}, ['path']),
                 view_image, 'visual'),
        ToolSpec('render_pdf', 'Render selected PDF pages and send native images to the model for layout/figure/equation inspection. '
                 'Use read_document for text. Maximum 6 pages per call, page numbers start at 1. Requires vision-capable model.',
                 params({'path': {'type': 'string'}, 'pages': {'type': 'array', 'items': {'type': 'integer'},
                         'minItems': 1, 'maxItems': 6, 'description': '1-based pages; default [1].'},
                         'dpi': {'type': 'integer', 'minimum': 72, 'maximum': 200,
                                 'description': 'Default 120; reduce for large-format pages.'}}, ['path']),
                 render_pdf, 'visual'),
        ToolSpec('start_job', 'Start a long-running PowerShell (Windows) or bash command and return immediately with a job_id. '
                 'Logs persist; jobs survive chat disconnect and server restart. Inspect with job_status/read_job_log; cancel explicitly. '
                 'Runs with user privileges, NOT sandboxed. Non-interactive; no stdin. Use it for commands expected to run longer than about a minute (long builds, simulations, servers); for tests and scripts that finish quickly use run_command, whose result you get in the same step',
                 params({'command': {'type': 'string'}, 'cwd': {'type': 'string'},
                         'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 604800,
                                     'description': 'Seconds, default 3600. Timeout terminates process tree.'}}, ['command']),
                 start_job, 'jobs', dangerous=True, needs_ctx=True),
        ToolSpec('job_status', 'Inspect a managed job: state, exit code, elapsed timestamps, bounded log sizes and truncation. '
                 'A lost worker means execution is uncertain; inspect side effects before rerunning.',
                 params({'job_id': {'type': 'string'}}, ['job_id']), job_status, 'jobs', needs_ctx=True),
        ToolSpec('job_wait', 'Block until a managed job finishes or `seconds` elapse (default 120, max 600), then return its '
                 'state, exit code and the tail of stdout/stderr. Use this instead of sleeping in run_command or polling job_status.',
                 params({'job_id': {'type': 'string'},
                         'seconds': {'type': 'integer', 'minimum': 1, 'maximum': 600},
                         'tail_bytes': {'type': 'integer', 'minimum': 0, 'maximum': 20000,
                                        'description': 'Bytes of each log tail to include (default 2000).'}}, ['job_id']),
                 job_wait, 'jobs', needs_ctx=True),
        ToolSpec('list_jobs', 'List the 30 most recent managed jobs belonging to this chat; recover handles after compaction/restart.',
                 params({}, []), list_jobs, 'jobs', needs_ctx=True),
        ToolSpec('read_job_log', 'Read a bounded chunk of a managed job stdout/stderr using a byte cursor. '
                 'Use next_cursor for the next chunk. at_end is current EOF, not necessarily job completion. '
                 'Logs cap at 16 MiB per stream; dropped bytes reported. UTF-8 decoding uses replacement.',
                 params({'job_id': {'type': 'string'}, 'stream': {'type': 'string', 'enum': ['stdout', 'stderr']},
                         'cursor': {'type': 'integer', 'minimum': 0},
                         'max_bytes': {'type': 'integer', 'minimum': 1, 'maximum': 200000}}, ['job_id']),
                 read_job_log, 'jobs', needs_ctx=True),
        ToolSpec('cancel_job', 'Request termination of a managed job process tree, retaining logs. Idempotent; '
                 'poll job_status to confirm cancellation. Chat Stop does not cancel detached jobs.',
                 params({'job_id': {'type': 'string'}}, ['job_id']), cancel_job, 'jobs', dangerous=True, needs_ctx=True),
    ]
    for spec in definitions:
        registry[spec.name] = spec
