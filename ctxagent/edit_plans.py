"""Reviewable hash-checked multi-file exact edits with durable backups/journal.

Per-file replace is atomic, NOT a filesystem-wide transaction. On failure the
journal describes partial application; never auto-overwrite concurrent edits.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from . import config
from .jobs import atomic_json


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def directory(identifier: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{32}', identifier):
        raise ValueError('Invalid edit plan ID')
    return config.DATA_DIR / 'edit_plans' / identifier


def atomic_bytes(path: Path, raw: bytes, *, expected_sha256: str | None = None) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.ctxedit')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(name, path.stat().st_mode)
        for attempt in range(6):
            # A Windows reader/scanner can briefly deny replacement. A retry
            # must recheck the source: another writer may have changed it.
            if expected_sha256 is not None and sha(path.read_bytes()) != expected_sha256:
                raise ValueError(f'File changed during atomic replacement: {path}')
            try:
                os.replace(name, path)
                break
            except PermissionError as exc:
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
    finally:
        Path(name).unlink(missing_ok=True)


def prepare(sb, owner: str, changes: list[dict], *, verbatim: bool = False) -> dict:
    if not 1 <= len(changes) <= 30:
        raise ValueError('Provide 1..30 file changes')
    identifier = uuid.uuid4().hex
    dest = directory(identifier)
    files, previews, seen, total = [], [], set(), 0
    for change in changes:
        p = sb.resolve(change['path'], must_exist=True)
        if p in seen:
            raise ValueError('Use one file entry with multiple edits per path')
        seen.add(p)
        raw = p.read_bytes()
        total += len(raw)
        if total > 8 * 1024 * 1024:
            raise ValueError('Plan sources exceed 8 MiB')
        if change.get('expected_sha256') and sha(raw) != change['expected_sha256']:
            raise ValueError(f'Expected hash mismatch: {p}')
        bom = raw.startswith(b'\xef\xbb\xbf')
        text = raw.decode('utf-8-sig' if bom else 'utf-8', errors='strict')
        if '\x00' in text:
            raise ValueError('Binary files are not supported')
        edits = change.get('edits') or []
        if not 1 <= len(edits) <= 100:
            raise ValueError('Each file needs 1..100 edits')
        after = text
        for edit in edits:
            old, new = edit['old_string'], edit['new_string']
            if not old:
                raise ValueError('old_string must not be empty')
            # Hand-authored patches inherit CRLF; native adapters already encode
            # exact records/cell strings and must not normalize embedded newlines.
            if not verbatim and '\r\n' in text and '\n' not in text.replace('\r\n', ''):
                old = old.replace('\r\n', '\n').replace('\n', '\r\n')
                new = new.replace('\r\n', '\n').replace('\n', '\r\n')
            count = after.count(old)
            if count == 0 or (count > 1 and not edit.get('replace_all', False)):
                raise ValueError(f'Edit matches {count} times in {p}; expected unique match')
            after = after.replace(old, new) if edit.get('replace_all', False) else after.replace(old, new, 1)
        output = after.encode('utf-8-sig' if bom else 'utf-8')
        if len(output) > 8 * 1024 * 1024:
            raise ValueError('Edited output exceeds 8 MiB')
        preview = ''.join(difflib.unified_diff(text.splitlines(keepends=True), after.splitlines(keepends=True),
                          fromfile=str(p), tofile=str(p)))
        previews.append(preview)
        files.append({'path': str(p), 'before_sha256': sha(raw), 'after_sha256': sha(output),
                      'before': raw, 'after': output})
    if sum(len(s) for s in previews) > 100000:
        raise ValueError('Preview exceeds 100000 characters; split the plan')
    dest.mkdir(parents=True)
    manifest = {'id': identifier, 'owner': owner, 'state': 'prepared', 'files': [], 'applied': []}
    for i, row in enumerate(files):
        (dest / f'{i}.before').write_bytes(row.pop('before'))
        (dest / f'{i}.after').write_bytes(row.pop('after'))
        manifest['files'].append(row)
    atomic_json(dest / 'plan.json', manifest)
    return {'plan_id': identifier, 'files': manifest['files'], 'diff': '\n'.join(previews),
            'next': 'Review the diff, then apply_edit_plan(plan_id). No source files changed yet.'}


def prepare_binary(sb, owner: str, path: str, expected_sha256: str, output: bytes,
                   preview: str, scope: dict) -> dict:
    """Stage a validated native-format candidate; the apply journal is shared."""
    p = sb.resolve(path, must_exist=True)
    raw = p.read_bytes()
    if len(raw) > 8 * 1024 * 1024 or len(output) > 8 * 1024 * 1024:
        raise ValueError('Native source/output exceeds 8 MiB.')
    if sha(raw) != expected_sha256:
        raise ValueError('Expected native source hash mismatch.')
    if raw == output or not preview or len(preview) > 100000:
        raise ValueError('Native plan needs a nonempty bounded change summary and changed output.')
    identifier = uuid.uuid4().hex
    dest = directory(identifier)
    dest.mkdir(parents=True)
    row = {'path': str(p), 'before_sha256': sha(raw), 'after_sha256': sha(output)}
    (dest / '0.before').write_bytes(raw)
    (dest / '0.after').write_bytes(output)
    manifest = {'id': identifier, 'owner': owner, 'state': 'prepared', 'files': [row],
                'applied': [], 'documents': [scope], 'native_format': p.suffix.lower()}
    atomic_json(dest / 'plan.json', manifest)
    return {'plan_id': identifier, 'files': [row], 'change_summary': preview,
            'next': 'Review the semantic change summary before applying. Source unchanged.'}


def apply(sb, owner: str, identifier: str, rollback: bool = False) -> dict:
    dest = directory(identifier)
    manifest = json.loads((dest / 'plan.json').read_text(encoding='utf-8'))
    if 'pending_index' in manifest:
        raise RuntimeError('Interrupted write has uncertain state; inspect pending_index and file hashes before recovery')
    if manifest['owner'] != owner:
        raise PermissionError('Edit plan belongs to another chat')
    if (dest / 'lock').exists():
        raise RuntimeError('Plan is locked; inspect journal before manual recovery')
    if not rollback and manifest['state'] != 'prepared':
        raise ValueError('Plan is not prepared; inspect its current state')
    if rollback and manifest['state'] not in ('applied', 'partial'):
        raise ValueError('Only applied/partial plans can be rolled back')
    indices = list(reversed(manifest['applied'])) if rollback else list(range(len(manifest['files'])))
    expected = 'after_sha256' if rollback else 'before_sha256'
    payload = 'before' if rollback else 'after'
    # All preconditions first, then recheck each immediately before replacing.
    for i in indices:
        row = manifest['files'][i]
        p = sb.resolve(row['path'], must_exist=True)
        if sha(p.read_bytes()) != row[expected]:
            raise ValueError(f'Stale file; refusing to overwrite: {p}')
        data = (dest / f'{i}.{payload}').read_bytes()
        if sha(data) != row['before_sha256' if rollback else 'after_sha256']:
            raise ValueError('Plan backup/output corrupted')
    lock = dest / 'lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        manifest['state'] = 'partial'
        atomic_json(dest / 'plan.json', manifest)
        for i in indices:
            row = manifest['files'][i]
            p = sb.resolve(row['path'], must_exist=True)
            if sha(p.read_bytes()) != row[expected]:
                raise ValueError(f'File changed during application: {p}')
            manifest['pending_index'] = i
            atomic_json(dest / 'plan.json', manifest)
            atomic_bytes(p, (dest / f'{i}.{payload}').read_bytes(), expected_sha256=row[expected])
            if rollback:
                manifest['applied'].remove(i)
            else:
                manifest['applied'].append(i)
            manifest.pop('pending_index', None)
            atomic_json(dest / 'plan.json', manifest)
        manifest['state'] = 'rolled_back' if rollback else 'applied'
        atomic_json(dest / 'plan.json', manifest)
    finally:
        lock.unlink(missing_ok=True)
    return {'plan_id': identifier, 'state': manifest['state'], 'files': manifest['files'],
            'journal': str(dest / 'plan.json'), 'note': 'Per-file atomic writes, not a global transaction.'}
