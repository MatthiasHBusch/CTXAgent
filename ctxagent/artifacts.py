"""Immutable PNG artifacts. Persist descriptors, expand bytes only at API boundary."""
from __future__ import annotations

import base64
import hashlib
import io
import math
import os
from pathlib import Path
import re
import tempfile

from . import config

MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PAGES = 6


def directory() -> Path:
    p = config.DATA_DIR / 'artifacts'
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_path(identifier: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{64}', identifier):
        raise ValueError('Invalid artifact ID')
    return directory() / (identifier + '.png')


def save_png(data: bytes, width: int, height: int, label: str, source_hash: str) -> dict:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Rendered image exceeds 12 MiB; reduce size/DPI')
    identifier = hashlib.sha256(data).hexdigest()
    target = artifact_path(identifier)
    if not target.exists():
        fd, name = tempfile.mkstemp(dir=target.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
    return {'id': identifier, 'mime': 'image/png', 'width': width, 'height': height,
            'bytes': len(data), 'label': label, 'source_sha256': source_hash,
            # Conservative provider-independent estimate, not billing accuracy.
            'estimated_tokens': max(2048, 256 * math.ceil(width / 512) * math.ceil(height / 512))}


def view(path: Path, max_side: int = 1600) -> list[dict]:
    from PIL import Image, ImageOps
    if not 256 <= max_side <= 2400:
        raise ValueError('max_side must be between 256 and 2400')
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Source image exceeds 50 MiB')
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as original:
        if original.width * original.height > MAX_PIXELS:
            raise ValueError('Source image exceeds 40 million pixels')
        original.seek(0)
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format='PNG')
        return [save_png(out.getvalue(), image.width, image.height,
                         f'{path.name} (first frame)', hashlib.sha256(raw).hexdigest())]


def render(path: Path, pages: list[int] | None = None, dpi: int = 120) -> list[dict]:
    import fitz
    pages = [1] if pages is None else pages
    if not pages or len(pages) > MAX_PAGES or any(type(p) is not int for p in pages):
        raise ValueError('Select 1 to 6 integer page numbers (1-based)')
    if not 72 <= dpi <= 200:
        raise ValueError('dpi must be between 72 and 200')
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('PDF exceeds 50 MiB')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with fitz.open(stream=raw, filetype='pdf') as doc:
        if doc.needs_pass:
            raise ValueError('Encrypted PDF requires a password; unlock it first')
        if any(p < 1 or p > len(doc) for p in pages):
            raise ValueError(f'Page numbers must be between 1 and {len(doc)}')
        # Preflight ALL requested pages before allocating any raster.
        for number in pages:
            rect = doc[number - 1].rect
            width, height = math.ceil(rect.width * dpi / 72), math.ceil(rect.height * dpi / 72)
            if width * height > 8_000_000 or max(width, height) > 4000:
                raise ValueError('Page raster too large; reduce DPI')
        out = []
        for number in dict.fromkeys(pages):
            pix = doc[number - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            desc = save_png(pix.tobytes('png'), pix.width, pix.height,
                            f'{path.name}, page {number}', digest)
            desc['page'] = number
            out.append(desc)
        return out


def expand_messages(messages: list[dict]) -> list[dict]:
    """Native user image blocks work across OpenRouter vision providers.

    Tool results remain text, followed by one synthetic user media message AFTER
    the entire result batch; never interrupt assistant/tool pairing with images.
    """
    out = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            out.append(message)
            continue
        blocks = []
        for block in content:
            if block.get('type') != 'image_ref':
                blocks.append(block)
                continue
            desc = block['image']
            data = artifact_path(desc['id']).read_bytes()
            if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != desc['id']:
                raise ValueError('Image artifact is oversized or corrupted')
            blocks.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii'),
                'detail': 'high'}, **{k: block[k] for k in ('cache_control', 'prompt_cache_breakpoint') if k in block}})
        out.append({**message, 'content': blocks})
    return out
