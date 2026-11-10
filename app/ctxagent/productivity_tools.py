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

    async def view_image(sb, path: str, max_side: int = 1600, canvas: bool = False, ctx=None):
        p = sb.resolve(path, must_exist=True)
        if canvas:
            if not ctx or not ctx.get('session'):
                raise ToolError('canvas=true requires a chat session')
            from .canvas import CanvasError, OPERATIONS
            try:
                desc = await asyncio.to_thread(ctx['session'].conv.canvas.open_file, p)
            except CanvasError as exc:
                raise ToolError(f'{exc}. Nothing was opened.')
            return result({'canvas_version': desc['version'], 'size': [desc['width'], desc['height']],
                           'operations': list(OPERATIONS),
                           'note': 'Opened on the live canvas: shown to the user in the Canvas panel and kept '
                                   'current at the end of each of your requests.'})
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
        ToolSpec('view_image', 'Show an image to the model, with a preview in the UI. Raster formats such as PNG, JPEG, WebP (first frame); not SVG. '
                 'canvas=true opens it on the live canvas instead (shown to the user, kept current, drawable).',
                 params({'path': {'type': 'string'}, 'max_side': {'type': 'integer', 'minimum': 256,
                         'maximum': 2400, 'description': 'Longest side limit; default 1600.'},
                         'canvas': {'type': 'boolean', 'description': 'Open on the live canvas (see canvas_open).'}},
                        ['path']),
                 view_image, 'visual', needs_ctx=True),
        ToolSpec('render_pdf', 'Render PDF pages as images for the model, up to 6 per call.',
                 params({'path': {'type': 'string'}, 'pages': {'type': 'array', 'items': {'type': 'integer'},
                         'minItems': 1, 'maxItems': 6, 'description': '1-based pages; default [1].'},
                         'dpi': {'type': 'integer', 'minimum': 72, 'maximum': 200,
                                 'description': 'Default 120; reduce for large-format pages.'}}, ['path']),
                 render_pdf, 'visual'),
        ToolSpec('start_job', 'Start a shell command (PowerShell on Windows, bash elsewhere) detached and return a job_id at once. The job survives chat Stop, disconnect and server restart; its logs are kept. stdin is closed.',
                 params({'command': {'type': 'string'}, 'cwd': {'type': 'string'},
                         'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 604800,
                                     'description': 'Seconds, default 3600. Timeout terminates process tree.'}}, ['command']),
                 start_job, 'jobs', dangerous=True, needs_ctx=True),
        ToolSpec('job_status', "State, exit code, timestamps and log sizes of a job. State 'lost' means it is unknown how far the command ran.",
                 params({'job_id': {'type': 'string'}}, ['job_id']), job_status, 'jobs', needs_ctx=True),
        ToolSpec('job_wait', 'Wait until a job finishes or `seconds` pass (default 120), then return state, exit code and the tail of stdout/stderr.',
                 params({'job_id': {'type': 'string'},
                         'seconds': {'type': 'integer', 'minimum': 1, 'maximum': 600},
                         'tail_bytes': {'type': 'integer', 'minimum': 0, 'maximum': 20000,
                                        'description': 'Bytes of each log tail to include (default 2000).'}}, ['job_id']),
                 job_wait, 'jobs', needs_ctx=True),
        ToolSpec('list_jobs', 'The 30 most recent jobs of this chat.',
                 params({}, []), list_jobs, 'jobs', needs_ctx=True),
        ToolSpec('read_job_log', "Read a chunk of a job's stdout or stderr from a byte cursor; next_cursor continues. Logs are capped at 16 MiB per stream.",
                 params({'job_id': {'type': 'string'}, 'stream': {'type': 'string', 'enum': ['stdout', 'stderr']},
                         'cursor': {'type': 'integer', 'minimum': 0},
                         'max_bytes': {'type': 'integer', 'minimum': 1, 'maximum': 200000}}, ['job_id']),
                 read_job_log, 'jobs', needs_ctx=True),
        ToolSpec('cancel_job', "Terminate a job's process tree; its logs are kept.",
                 params({'job_id': {'type': 'string'}}, ['job_id']), cancel_job, 'jobs', dangerous=True, needs_ctx=True),
    ]
    for spec in definitions:
        registry[spec.name] = spec
