"""Readers for document containers: PDF, DOCX, PPTX, Jupyter notebooks, ODT, EPUB.

Every dependency is optional and probed at call time; a missing library produces
a clear message naming the package to install rather than a traceback.

  PDF       level 0  pages + metadata + top-level bookmarks
            level 1  + full bookmark tree, + detected heading lines per page
            level 2  + first 2 lines of every page
            level 3+ + first paragraph of every page
  DOCX      level 0  title + counts;  1  H1-H2;  2  all headings + tables;  3+ first line under each
  PPTX      level 0  slide count + titles;  1+ + bullet text per slide
  notebook  level 0  kernel + cell counts + H1;  1  + markdown headings + first code line
            level 2  + code structure (classes/functions);  3+ + output summaries
"""

from __future__ import annotations

import io
import json
import re
import textwrap
from pathlib import Path

_HEADINGish = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z][^.!?]{2,80}$")


def _need(pkg: str, pip: str) -> str:
    return f"[`{pkg}` is not installed — run `pip install {pip}` to read this format]"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _pdf_open(path: Path):
    """Prefer PyMuPDF (better text order), fall back to pypdf."""
    try:
        import fitz  # PyMuPDF

        return "fitz", fitz.open(str(path))
    except Exception:
        pass
    try:
        from pypdf import PdfReader

        return "pypdf", PdfReader(str(path))
    except Exception:
        return None, None


def _pdf_pages_text(kind: str, doc) -> list[str]:
    if kind == "fitz":
        return [doc[i].get_text() for i in range(doc.page_count)]
    return [(p.extract_text() or "") for p in doc.pages]


def _pdf_outline_entries(kind: str, doc) -> list[tuple[int, str, int]]:
    """(depth, title, page) triples from the document bookmarks."""
    out: list[tuple[int, str, int]] = []
    if kind == "fitz":
        try:
            for lvl, title, page in doc.get_toc(simple=True):
                out.append((int(lvl), str(title).strip(), int(page)))
        except Exception:
            pass
    else:
        try:
            def walk(items, depth=1):
                for it in items:
                    if isinstance(it, list):
                        walk(it, depth + 1)
                    else:
                        try:
                            pg = doc.get_destination_page_number(it) + 1
                        except Exception:
                            pg = 0
                        out.append((depth, str(it.title).strip(), pg))
            walk(doc.outline)
        except Exception:
            pass
    return out


def _pdf_meta(kind: str, doc) -> dict:
    try:
        if kind == "fitz":
            return dict(doc.metadata or {})
        md = doc.metadata or {}
        return {k.lstrip("/").lower(): str(v) for k, v in md.items()}
    except Exception:
        return {}


def extract_pdf(path: Path) -> str:
    kind, doc = _pdf_open(path)
    if doc is None:
        return _need("PyMuPDF or pypdf", "pymupdf")
    pages = _pdf_pages_text(kind, doc)
    return "\n\n".join(f"--- page {i + 1} ---\n{t}" for i, t in enumerate(pages))


def outline_pdf(path: Path, level: int) -> str:
    kind, doc = _pdf_open(path)
    if doc is None:
        return _need("PyMuPDF or pypdf", "pymupdf")
    n_pages = doc.page_count if kind == "fitz" else len(doc.pages)
    meta = _pdf_meta(kind, doc)
    out = [f"{n_pages} pages"]
    for key in ("title", "author", "subject", "creator", "producer"):
        v = meta.get(key) or meta.get(key.capitalize())
        if v and str(v).strip():
            out.append(f"{key}: {str(v).strip()[:160]}")

    toc = _pdf_outline_entries(kind, doc)
    if toc:
        max_depth = 1 if level == 0 else 99
        out.append(f"--- bookmarks ({len(toc)}) ---")
        shown = 0
        for depth, title, page in toc:
            if depth <= max_depth:
                out.append("  " * (depth - 1) + f"{title}   (p.{page})")
                shown += 1
        if shown < len(toc):
            out.append(f"… {len(toc) - shown} deeper bookmarks hidden at this level")
    else:
        out.append("(no bookmarks in this PDF)")

    if level >= 1:
        pages = _pdf_pages_text(kind, doc)
        if level == 1 and not toc:
            out.append("--- lines that look like headings ---")
            for i, t in enumerate(pages[:200], 1):
                for ln in t.splitlines()[:40]:
                    s = ln.strip()
                    if 3 < len(s) < 80 and _HEADINGish.match(s):
                        out.append(f"  p.{i}: {s}")
        if level >= 2:
            n_lines = 2 if level == 2 else 6
            out.append(f"--- first {n_lines} line(s) of each page ---")
            for i, t in enumerate(pages[:400], 1):
                head = [x.strip() for x in t.splitlines() if x.strip()][:n_lines]
                if head:
                    out.append(f"  p.{i}: " + " / ".join(head)[:300])
        words = sum(len(t.split()) for t in pages)
        out.append("")
        out.append(f"[{words:,} words extracted]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _docx(path: Path):
    try:
        import docx

        return docx.Document(str(path))
    except ImportError:
        return None
    except Exception:
        return None


def extract_docx(path: Path) -> str:
    d = _docx(path)
    if d is None:
        return _need("python-docx", "python-docx")
    parts = [p.text for p in d.paragraphs]
    for i, tbl in enumerate(d.tables, 1):
        parts.append(f"--- table {i} ---")
        for row in tbl.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def outline_docx(path: Path, level: int) -> str:
    d = _docx(path)
    if d is None:
        return _need("python-docx", "python-docx")
    heads: list[tuple[int, str, int]] = []
    body_paras = 0
    for i, p in enumerate(d.paragraphs):
        style = (p.style.name or "") if p.style is not None else ""
        m = re.match(r"Heading (\d)", style)
        if m and p.text.strip():
            heads.append((int(m.group(1)), p.text.strip(), i))
        elif p.text.strip():
            body_paras += 1

    out = []
    if heads:
        out.append(f"title: {heads[0][1]}")
    out.append(f"{len(d.paragraphs)} paragraphs ({body_paras} non-heading), "
               f"{len(d.tables)} tables, {len(heads)} headings")

    max_depth = 1 if level == 0 else (2 if level == 1 else 9)
    if heads:
        out.append(f"--- headings (depth ≤ {max_depth}) ---")
        shown = 0
        for depth, text, idx in heads:
            if depth <= max_depth:
                out.append("  " * (depth - 1) + f"{'#' * depth} {text}")
                shown += 1
                if level >= 3:
                    nxt = next((p.text.strip() for p in d.paragraphs[idx + 1: idx + 5]
                                if p.text.strip()), "")
                    if nxt:
                        out.append("  " * depth + textwrap.shorten(nxt, 180, placeholder=" …"))
        if shown < len(heads):
            out.append(f"… {len(heads) - shown} deeper heading(s) hidden at this level")

    if level >= 2 and d.tables:
        out.append("--- tables ---")
        for i, t in enumerate(d.tables, 1):
            hdr = " | ".join(c.text.strip()[:24] for c in t.rows[0].cells) if t.rows else ""
            out.append(f"  table {i}: {len(t.rows)} rows x {len(t.columns)} cols   {hdr}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _pptx(path: Path):
    try:
        from pptx import Presentation

        return Presentation(str(path))
    except Exception:
        return None


def extract_pptx(path: Path) -> str:
    pr = _pptx(path)
    if pr is None:
        return _need("python-pptx", "python-pptx")
    parts = []
    for i, slide in enumerate(pr.slides, 1):
        parts.append(f"--- slide {i} ---")
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                parts.append(sh.text_frame.text)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("[notes] " + slide.notes_slide.notes_text_frame.text)
    return "\n".join(parts)


def outline_pptx(path: Path, level: int) -> str:
    pr = _pptx(path)
    if pr is None:
        return _need("python-pptx", "python-pptx")
    out = [f"{len(pr.slides)} slides"]
    for i, slide in enumerate(pr.slides, 1):
        title = ""
        try:
            if slide.shapes.title is not None:
                title = slide.shapes.title.text.strip()
        except Exception:
            pass
        out.append(f"  slide {i}: {title or '(no title)'}")
        if level >= 1:
            for sh in slide.shapes:
                if not sh.has_text_frame:
                    continue
                t = sh.text_frame.text.strip()
                if t and t != title:
                    for ln in t.splitlines()[: 3 if level == 1 else 20]:
                        if ln.strip():
                            out.append("      • " + ln.strip()[:160])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Jupyter notebooks
# ---------------------------------------------------------------------------

def _nb(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _src(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def extract_notebook(path: Path) -> str:
    nb = _nb(path)
    if nb is None:
        return "[could not parse notebook]"
    parts = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        ct = cell.get("cell_type")
        parts.append(f"--- cell {i} [{ct}] ---")
        parts.append(_src(cell))
        for o in cell.get("outputs", [])[:3]:
            txt = o.get("text") or (o.get("data", {}) or {}).get("text/plain") or ""
            if isinstance(txt, list):
                txt = "".join(txt)
            if txt:
                parts.append("[out] " + str(txt)[:2000])
    return "\n".join(parts)


def outline_notebook(path: Path, level: int) -> str:
    nb = _nb(path)
    if nb is None:
        return "[could not parse notebook]"
    cells = nb.get("cells", [])
    kern = ((nb.get("metadata") or {}).get("kernelspec") or {}).get("display_name", "?")
    n_md = sum(1 for c in cells if c.get("cell_type") == "markdown")
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    out = [f"kernel: {kern}", f"{len(cells)} cells ({n_md} markdown, {n_code} code)"]

    heads = []
    for c in cells:
        if c.get("cell_type") == "markdown":
            for ln in _src(c).splitlines():
                m = re.match(r"^(#{1,6})\s+(.*)", ln.strip())
                if m:
                    heads.append((len(m.group(1)), m.group(2).strip()))
    max_depth = 1 if level == 0 else level + 1
    if heads:
        out.append(f"--- markdown headings (depth ≤ {max_depth}) ---")
        for d, t in heads:
            if d <= max_depth:
                out.append("  " * (d - 1) + f"{'#' * d} {t}")

    if level >= 1:
        out.append("--- code cells ---")
        for i, c in enumerate(cells, 1):
            if c.get("cell_type") != "code":
                continue
            src = _src(c)
            first = next((l for l in src.splitlines() if l.strip()), "")
            out.append(f"  [{i}] {first[:140]}")
        if level >= 2:
            from . import code as codemod

            all_code = "\n\n".join(_src(c) for c in cells if c.get("cell_type") == "code")
            out.append("--- code structure across all cells ---")
            out.append(codemod.outline_python(all_code, min(level, 4)))
    if level >= 3:
        out.append("--- outputs ---")
        for i, c in enumerate(cells, 1):
            outs = c.get("outputs") or []
            if outs:
                kinds = ", ".join(o.get("output_type", "?") for o in outs)
                out.append(f"  [{i}] {len(outs)} output(s): {kinds}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def extract(path: Path, kind: str) -> str:
    if kind == "pdf":
        return extract_pdf(path)
    if kind == "docx":
        return extract_docx(path)
    if kind == "pptx":
        return extract_pptx(path)
    if kind == "notebook":
        return extract_notebook(path)
    if kind in ("odt", "epub"):
        return _need("a converter", "odfpy  # or convert to docx/pdf first")
    return "[unsupported document type]"


def outline(path: Path, level: int, kind: str) -> str:
    if kind == "pdf":
        return outline_pdf(path, level)
    if kind == "docx":
        return outline_docx(path, level)
    if kind == "pptx":
        return outline_pptx(path, level)
    if kind == "notebook":
        return outline_notebook(path, level)
    return "[unsupported document type]"
