"""Conservative observation coverage. Missing metadata never means whole-file.

A version is the SHA-256 of source bytes, not mtime (external writers may preserve
mtime). Representations distinguish extracted lines from source lines and folded
views from outlines. Invalid/truncated views cannot replace a complete one.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def observation(path: Path, representation: str, coverage: dict, *, complete=True, source_version: str | None = None) -> dict:
    after = version(path)
    unchanged = source_version is None or source_version == after
    return {'version': after, 'representation': representation,
            'coverage': coverage, 'complete': bool(complete and unchanged)}


def covers(new: dict, old: dict) -> bool:
    if not new or not old or not new.get('complete') or not old.get('complete'):
        return False
    if not new.get('version') or new.get('version') != old.get('version'):
        return False
    if new.get('representation') != old.get('representation'):
        return False
    a, b = new.get('coverage', {}), old.get('coverage', {})
    if a.get('kind') == 'whole':
        return True
    if a.get('kind') != b.get('kind'):
        return False
    if a.get('kind') == 'lines':
        return a['start'] <= b['start'] and a['end'] >= b['end']
    if a.get('kind') == 'selectors':
        # Exact selector containment only; do not guess semantic relationships.
        return set(a.get('values', ())) >= set(b.get('values', ()))
    return False
