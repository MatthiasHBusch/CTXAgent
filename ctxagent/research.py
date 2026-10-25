"""Scholarly metadata and locally verified evidence records.

Crossref metadata is discovery data, not evidence for a paper's claims. Quotation
verification establishes textual presence only, not correctness or entailment.
Network calls are fixed to Crossref, bounded, and carry no local document text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
import uuid

import httpx

from . import readers
from .jobs import atomic_json
from .observations import version

MAX_RESPONSE = 4 * 1024 * 1024


def doi_value(value: str) -> str:
    doi = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value.strip(), flags=re.I)
    if not re.fullmatch(r'10\.\d{4,9}/\S+', doi) or len(doi) > 512:
        raise ValueError('Provide a valid DOI or doi.org URL')
    return doi


async def request_crossref(endpoint: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        async with client.stream('GET', 'https://api.crossref.org/works' + endpoint,
                                 params=params, headers={'User-Agent': 'ctxagent-research/1.0 (local scholarly assistant)'}) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_RESPONSE:
                    raise ValueError('Crossref response exceeds 4 MiB')
    payload = json.loads(data)
    if payload.get('status') != 'ok':
        raise ValueError('Crossref returned an unsuccessful response')
    return payload['message']


def metadata(item: dict) -> dict:
    parts = (item.get('published') or item.get('issued') or {}).get('date-parts') or []
    return {'doi': item.get('DOI'), 'title': (item.get('title') or [''])[0],
            'authors': [{'given': a.get('given', ''), 'family': a.get('family', ''), 'ORCID': a.get('ORCID')}
                        for a in item.get('author', [])[:100]],
            'year': parts[0][0] if parts and parts[0] else None,
            'container': (item.get('container-title') or [''])[0], 'type': item.get('type'),
            'volume': item.get('volume'), 'issue': item.get('issue'), 'pages': item.get('page'),
            'publisher': item.get('publisher'), 'url': item.get('URL'),
            'relations': item.get('relation') or {}, 'updates': item.get('update-to') or [],
            'provenance': {'service': 'Crossref', 'retrieved_at': time.time()},
            'note': 'Publisher-deposited metadata, not independently verified scientific evidence.'}


async def search(query: str, limit: int = 5) -> dict:
    if not query.strip() or len(query) > 1000 or not 1 <= limit <= 20:
        raise ValueError('query must be 1..1000 characters; limit 1..20')
    data = await request_crossref('', {'query.bibliographic': query, 'rows': limit})
    return {'query': query, 'total_results': data.get('total-results'),
            'results': [metadata(item) for item in data.get('items', [])],
            'note': 'Search hits are candidates; verify DOI/title/authors before citing.'}


async def lookup(doi: str) -> dict:
    normalized = doi_value(doi)
    data = await request_crossref('/' + quote(normalized, safe=''))
    result = metadata(data)
    if (result.get('doi') or '').lower() != normalized.lower():
        raise ValueError('Returned DOI does not match requested DOI')
    return result


def bibtex(item: dict, key: str) -> str:
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_:-]{0,100}', key):
        raise ValueError('Citation key must start with a letter and use only letters, digits, _, :, -')
    escape = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '%': r'\%', '&': r'\&',
              '_': r'\_', '#': r'\#', '$': r'\$', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    def safe(value):
        return ''.join(escape.get(c, c) for c in str(value).replace('\n', ' '))
    fields = {'title': '{' + safe(item.get('title', '')) + '}',
              'author': ' and '.join(safe(a['family']) + (', ' + safe(a['given']) if a['given'] else '')
                                     for a in item.get('authors', [])),
              'year': item.get('year'), 'doi': doi_value(item['doi']),
              'journal': safe(item.get('container', '')), 'volume': safe(item.get('volume') or ''),
              'number': safe(item.get('issue') or ''), 'pages': safe(item.get('pages') or '')}
    # DOI backslashes/braces are uncommon but still data, never TeX instructions.
    fields['doi'] = safe(fields['doi'])
    kind = 'article' if item.get('type') == 'journal-article' else 'misc'
    return '@' + kind + '{' + key + ',\n' + ''.join('  ' + k + ' = {' + str(v) + '},\n'
                for k, v in fields.items() if v not in (None, '')) + '}\n'


def record_evidence(sb, source: str, quotation: str, claim: str, output_dir: str,
                    page: int | None = None, doi: str | None = None) -> dict:
    p = sb.resolve(source, must_exist=True)
    destination = sb.resolve(output_dir)
    if not quotation.strip() or len(quotation) > 20000 or not claim.strip() or len(claim) > 20000:
        raise ValueError('Quotation and claim must be nonempty and at most 20000 characters')
    if p.stat().st_size > 50 * 1024 * 1024:
        raise ValueError('Source exceeds 50 MiB')
    before = version(p)
    if p.suffix.lower() == '.pdf':
        import fitz
        if type(page) is not int:
            raise ValueError('PDF evidence requires an explicit 1-based page')
        with fitz.open(p) as doc:
            if not 1 <= page <= len(doc):
                raise ValueError('Page outside PDF')
            text = doc[page - 1].get_text()
        locator = {'page': page}
    else:
        if page is not None:
            raise ValueError('page is only supported for PDFs')
        text = readers.extract_text(p)
        locator = {'representation': 'extracted text'}
    norm = lambda value: ' '.join(value.split())
    haystack, needle = norm(text), norm(quotation)
    position = haystack.find(needle)
    if position < 0:
        raise ValueError('Quotation not found after whitespace normalization; no evidence record written')
    if version(p) != before:
        raise ValueError('Source changed during extraction')
    identifier = uuid.uuid4().hex
    record = {'id': identifier, 'source_path': str(p), 'source_sha256': before,
              'doi': doi_value(doi) if doi else None, 'locator': locator,
              'quotation': quotation, 'claim': claim, 'created_at': time.time(),
              'verification': {'method': 'exact text after whitespace normalization',
                  'occurrences': haystack.count(needle), 'normalized_offset': position,
                  'context_excerpt': haystack[max(0,position-160):position+len(needle)+160]},
              'status': 'text_presence_verified; claim_support_requires_review',
              'limitations': 'Does not establish entailment, scientific validity, or that this DOI belongs to this file.'}
    destination.mkdir(parents=True, exist_ok=True)
    target = sb.resolve(str(destination / (identifier + '.evidence.json')))
    atomic_json(target, record)
    return {'record_path': str(target), **record}
