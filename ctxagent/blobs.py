"""Immutable, atomically written blobs. Legacy 16-hex blob handles remain readable."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile

from .config import BLOB_DIR, ensure_dirs


def _path(blob_id: str) -> Path:
    if not re.fullmatch(r'(?:[a-f0-9]{16}|[a-f0-9]{64})', blob_id):
        raise ValueError('Invalid blob ID')
    return BLOB_DIR / f'{blob_id}.txt'


def put_blob(session_id: str, segment_id: str, content: str) -> str:
    ensure_dirs()
    raw = content.encode('utf-8')
    digest = hashlib.sha256(session_id.encode('utf-8') + b'\0' +
                            segment_id.encode('utf-8') + b'\0' + raw).hexdigest()
    target = _path(digest)
    if not target.exists():
        fd, name = tempfile.mkstemp(dir=target.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
    return digest


def get_blob(blob_id: str) -> str | None:
    p = _path(blob_id)
    if not p.exists():
        return None
    return p.read_bytes().decode('utf-8')


def blob_size(blob_id: str) -> int:
    p = _path(blob_id)
    return p.stat().st_size if p.exists() else 0
