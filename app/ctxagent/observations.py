"""Observation metadata recorded with a read result: source version, representation,
coverage, completeness.

A version is the SHA-256 of source bytes, not mtime (external writers may preserve
mtime). Representations distinguish extracted lines from source lines and folded
views from outlines.
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
