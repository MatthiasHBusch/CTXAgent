"""Durable jobs supervised by an independent Python worker, not an SSE task.

The worker owns its child process and watches a cancellation marker. No server
restart ever kills an arbitrary reused PID or automatically reruns a command.
Commands use the user's privileges: cwd confinement is NOT a process sandbox.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

from . import config

TERMINAL = {'succeeded', 'failed', 'cancelled', 'timed_out', 'lost'}
MAX_LOG_BYTES = 16 * 1024 * 1024  # per stream; excess is drained and counted


def atomic_json(path: Path, data: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/antivirus can briefly deny replacement of an open
        # destination. Retry the metadata rename, NEVER the user's command.
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.01 * 2 ** attempt, 0.2))
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    # Windows atomic rename/read sharing can briefly deny access. Retry the
    # metadata read only; never retry the user's command.
    for attempt in range(8):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.01 * 2 ** attempt, 0.2))


def directory() -> Path:
    p = config.DATA_DIR / 'jobs'
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Invalid job ID')
    return directory() / job_id


def shell_args(command: str) -> list[str]:
    import shutil
    if sys.platform == 'win32':
        shell = shutil.which('pwsh') or shutil.which('powershell')
        if not shell:
            raise RuntimeError('PowerShell is not installed')
        return [shell, '-NoProfile', '-NonInteractive', '-Command', command]
    return ['/bin/bash', '-lc', command]


def start(command: str, cwd: Path, owner: str, timeout: int = 3600, publish=None,
          preserve_managed_children: bool = False) -> dict:
    if not command.strip() or len(command) > 100_000:
        raise ValueError('Command must contain 1 to 100000 characters')
    if not 1 <= timeout <= 7 * 86400:
        raise ValueError('timeout must be 1 second to 7 days')
    if not cwd.is_dir():
        raise ValueError('cwd must be a directory')
    # Fail before creating records if the process-management dependency is absent.
    import psutil  # noqa: F401
    shell_args(command)
    identifier = uuid.uuid4().hex
    dest = job_dir(identifier)
    dest.mkdir()
    request = {'id': identifier, 'owner': owner, 'command': command, 'cwd': str(cwd),
               'timeout': timeout, 'created_at': time.time(), 'python': sys.executable,
               'platform': sys.platform, 'log_limit_bytes': MAX_LOG_BYTES,
               'preserve_managed_children': preserve_managed_children}
    atomic_json(dest / 'request.json', request)
    atomic_json(dest / 'status.json', {**request, 'state': 'starting'})
    (dest / 'stdout.log').touch()
    (dest / 'stderr.log').touch()
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' else 0
    try:
        if publish is not None:
            publish(identifier)
        # An absolute worker script and explicit app path avoid dependence on cwd/PYTHONPATH.
        with (dest / 'worker.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('job_worker.py')),
                                     str(dest.resolve())], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, close_fds=True,
                                    creationflags=flags, start_new_session=os.name != 'nt')
        # Retain/reap workers to avoid zombies on POSIX; workers survive server exit.
        import threading
        threading.Thread(target=proc.wait, daemon=True).start()
    except Exception as exc:
        atomic_json(dest / 'status.json', {**request, 'state': 'failed', 'error': str(exc),
                                          'finished_at': time.time()})
        raise
    return {**request, 'state': 'starting'}


def status(job_id: str, owner: str) -> dict:
    dest = job_dir(job_id)
    request_path = dest / 'request.json'
    if not request_path.exists():
        raise ValueError('No such job')
    request = read_json(request_path)
    if request['owner'] != owner:
        raise PermissionError('Job belongs to another chat')
    data = read_json(dest / 'status.json')
    if data['state'] not in TERMINAL:
        import psutil
        pid = data.get('worker_pid')
        alive = False
        if pid:
            try:
                p = psutil.Process(pid)
                alive = abs(p.create_time() - data['worker_created']) < 0.1 and p.is_running()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if (pid and not alive) or (not pid and time.time() - request['created_at'] > 30):
            # Do not overwrite worker metadata, or kill anything based on a stale PID.
            data = {**data, 'state': 'lost', 'error': 'Worker unavailable; inspect side effects before retrying.'}
    data['log_bytes'] = {name: (dest / (name + '.log')).stat().st_size
                         for name in ('stdout', 'stderr')}
    data['cancel_requested'] = (dest / 'cancel').exists()
    data['log_directory'] = str(dest)
    return data


def cancel(job_id: str, owner: str) -> dict:
    data = status(job_id, owner)
    if data['state'] not in TERMINAL:
        (job_dir(job_id) / 'cancel').touch()
        data['cancel_requested'] = True
    return data


def read_log(job_id: str, owner: str, stream: str = 'stdout', cursor: int = 0,
             max_bytes: int = 16000) -> dict:
    data = status(job_id, owner)
    if stream not in ('stdout', 'stderr'):
        raise ValueError('stream must be stdout or stderr')
    if cursor < 0 or not 1 <= max_bytes <= 200_000:
        raise ValueError('cursor must be >=0 and max_bytes must be 1..200000')
    p = job_dir(job_id) / (stream + '.log')
    size = p.stat().st_size
    if cursor > size:
        raise ValueError(f'cursor exceeds log size ({size})')
    with p.open('rb') as f:
        f.seek(cursor)
        raw = f.read(max_bytes)
    return {'job_id': job_id, 'stream': stream, 'cursor': cursor,
            'next_cursor': cursor + len(raw), 'size': size,
            'text': raw.decode('utf-8', errors='replace'),
            'encoding': 'utf-8 with replacement; byte cursors may split multibyte characters',
            'state': data['state'], 'at_end': cursor + len(raw) >= size,
            'dropped_bytes': data.get('dropped_bytes', {}).get(stream, 0)}


def list_owned(owner: str, limit: int = 30) -> list[dict]:
    out = []
    for p in sorted(directory().glob('*/request.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if json.loads(p.read_text(encoding='utf-8')).get('owner') == owner:
                out.append(status(p.parent.name, owner))
        except (OSError, ValueError):
            continue
        if len(out) >= limit:
            break
    return out
