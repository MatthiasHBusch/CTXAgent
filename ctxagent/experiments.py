"""Explicit, local reproducibility manifests. No hidden kernel state or environment dump."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from . import jobs
from .observations import version


def environment() -> dict:
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'sympy', 'pint', 'pytest'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': versions, 'note': 'Selected packages only; no environment variables/secrets captured.'}


def file_records(sb, paths: list[str], allow_missing=False) -> list[dict]:
    if len(paths) > 200:
        raise ValueError('At most 200 explicit file paths')
    out = []
    for name in paths:
        p = sb.resolve(name)
        if not p.is_file():
            if allow_missing:
                out.append({'path': str(p), 'exists': False})
                continue
            raise ValueError(f'Not a regular file: {p}')
        if p.stat().st_size > 2 * 1024**3:
            raise ValueError(f'File over 2 GiB; supply an external dataset identifier instead: {p}')
        before = p.stat()
        digest = version(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'File changed while hashing: {p}')
        out.append({'path': str(p), 'exists': True, 'sha256': digest, 'bytes': after.st_size})
    return out


def git_state(cwd: Path) -> dict:
    try:
        def run(*args):
            r = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, timeout=10)
            return r.returncode, r.stdout.decode('utf-8', 'replace').strip()
        rc, commit = run('rev-parse', 'HEAD')
        if rc:
            return {'available': False}
        rc, changes = run('status', '--porcelain')
        return {'available': True, 'commit': commit, 'dirty': bool(changes),
                'changes': changes[:20000], 'status_truncated': len(changes) > 20000}
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False}


def save_new(path: Path, record: dict) -> None:
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.manifest-tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link fails if destination exists; no window exposing a partial JSON file.
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def create(sb, output: str, cwd: str, command: str, code_files: list[str], inputs: list[str],
           parameters: dict, seeds: dict) -> dict:
    wd = sb.resolve(cwd, must_exist=True)
    target = sb.resolve(output)
    if not wd.is_dir() or len(command) > 100000:
        raise ValueError('cwd must be a directory and command at most 100000 characters')
    # Validate parameters before any writes. Recording seeds does not apply them.
    if len(json.dumps({'parameters': parameters, 'seeds': seeds}, allow_nan=False)) > 100000:
        raise ValueError('Parameter metadata exceeds 100000 characters')
    record = {'schema': 'ctxagent.experiment.v1', 'id': uuid.uuid4().hex, 'created_at': time.time(),
              'phase': 'planned', 'cwd': str(wd), 'command': command,
              'code': file_records(sb, code_files), 'inputs': file_records(sb, inputs),
              'parameters': parameters, 'seeds': seeds, 'environment': environment(),
              'git': git_state(wd), 'limitations': 'Manifest records declared inputs/code/seeds; '
              'does not prove completeness or set RNG seeds. Use clean process execution and verify outputs.'}
    save_new(target, record)
    return {'manifest_path': str(target), **record}


def finalize(sb, manifest: str, output: str, outputs: list[str], owner: str, job_id: str | None = None) -> dict:
    source = sb.resolve(manifest, must_exist=True)
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Manifest exceeds 2 MiB')
    record = json.loads(source.read_text(encoding='utf-8'))
    if record.get('schema') != 'ctxagent.experiment.v1' or record.get('phase') != 'planned':
        raise ValueError('Expected a planned experiment manifest')
    status = jobs.status(job_id, owner) if job_id else None
    if status and status['state'] not in jobs.TERMINAL:
        raise ValueError('Job still active; wait before finalizing')
    original_files = record['code'] + record['inputs']
    current = file_records(sb, [row['path'] for row in original_files], allow_missing=True)
    drift = [{'path': old['path'], 'before': old.get('sha256'), 'after': new.get('sha256')}
             for old, new in zip(original_files, current) if old.get('sha256') != new.get('sha256')]
    artifacts = file_records(sb, outputs, allow_missing=True)
    result = {**record, 'phase': 'recorded', 'recorded_at': time.time(),
              'planned_manifest_sha256': version(source), 'job': status,
              'command_matches_job': status['command'] == record['command'] if status else None,
              'cwd_matches_job': os.path.normcase(status['cwd']) == os.path.normcase(record['cwd']) if status else None,
              'input_or_code_drift': drift, 'outputs': artifacts,
              'missing_outputs': [row['path'] for row in artifacts if not row['exists']],
              'job_success_and_declared_inputs_unchanged': bool(status and status['state'] == 'succeeded'
                                         and status['command'] == record['command']
                                         and os.path.normcase(status['cwd']) == os.path.normcase(record['cwd'])
                                         and not drift and all(row['exists'] for row in artifacts)),
              'output_provenance_verified': False,
              'output_provenance_note': 'Output hashes record current files; they do not prove this job created them. Clean-run comparison is required.'}
    target = sb.resolve(output)
    save_new(target, result)
    return {'manifest_path': str(target), **result}
