"""Document readers: full-text extraction and level-of-detail outlines.

Two orthogonal operations on any supported file:

  extract_text(path)        -> the readable content, whatever the container
  outline(path, level)      -> a *static* structural summary at a detail level

`outline` is the cheap half of "give the agent a summary". It is deterministic,
instant, free, and — importantly — never hallucinates, because it only ever
reports structure that is literally present in the file. The LLM half lives in
`readers/llm.py`.

Two different meanings of "level", deliberately
-----------------------------------------------
**Static outlines take a structural level**: 0 is the barest skeleton and each
step adds one layer of internal detail. What a layer *is* depends on the format,
which is the whole point of having per-format readers:

  level  code                     markdown/latex   tabular            structured
  -----------------------------------------------------------------------------
  0      header + class names     title, top heads shape + columns    top keys
  1      + function/method names  + one more depth + dtypes, index    + depth 2
  2      + full signatures        + one more depth + per-column stats + depth 3
  3      + docstring first lines  + one more depth + first rows       + depth 4
  4      + full docstrings        + one more depth + more rows        + depth 5
  5      the document's full text

Static outlining is deterministic and free, so there is nothing to ration: the
level says *what kind* of detail you want, and the size follows from the file.
A `MAX_OUTLINE_CHARS` backstop keeps a pathological file from flooding context,
but it is a safety net, not a budget.

**LLM summaries take a token budget** instead -- 100 / 300 / 1,000 / 3,000 /
10,000 / uncapped -- because there the length is a real choice with a real price,
and a model will otherwise write as much as it likes. See `readers/llm.py`.

Tier renders are cached on (path, mtime, level), so a folder survey that outlines
the same file at two levels parses it once.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable

from .. import tokens
from . import code, docs, latex, markup, structured, tabular

MAX_OUTLINE_CHARS = 120_000
MAX_TEXT_CHARS = 400_000

# extension -> (kind, extractor, outliner)
_CODE_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "c-like", ".jsx": "c-like", ".ts": "c-like", ".tsx": "c-like",
    ".java": "c-like", ".c": "c-like", ".h": "c-like", ".cpp": "c-like",
    ".cc": "c-like", ".hpp": "c-like", ".cs": "c-like", ".go": "c-like",
    ".rs": "c-like", ".swift": "c-like", ".kt": "c-like", ".scala": "c-like",
    ".php": "c-like", ".rb": "ruby", ".m": "matlab", ".jl": "julia",
    ".r": "r", ".R": "r", ".sh": "shell", ".ps1": "shell", ".sql": "sql",
}
_MARKUP_EXT = {".md": "markdown", ".markdown": "markdown", ".rst": "rst",
               ".html": "html", ".htm": "html", ".tex": "latex", ".txt": "text",
               ".log": "text", ".rtf": "text"}
_TABULAR_EXT = {".csv": "csv", ".tsv": "tsv", ".xlsx": "excel", ".xlsm": "excel",
                ".xls": "excel", ".parquet": "parquet"}
_DOC_EXT = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".ipynb": "notebook",
            ".odt": "odt", ".epub": "epub"}
_STRUCT_EXT = {".json": "json", ".jsonl": "jsonl", ".yaml": "yaml", ".yml": "yaml",
               ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".xml": "xml"}


def detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _CODE_EXT:
        return _CODE_EXT[ext]
    if ext in _MARKUP_EXT:
        return _MARKUP_EXT[ext]
    if ext in _TABULAR_EXT:
        return _TABULAR_EXT[ext]
    if ext in _DOC_EXT:
        return _DOC_EXT[ext]
    if ext in _STRUCT_EXT:
        return _STRUCT_EXT[ext]
    if path.name.lower() in ("makefile", "dockerfile", "license", "readme"):
        return "text"
    return "unknown"


SUPPORTED_EXTENSIONS = sorted(
    set(_CODE_EXT) | set(_MARKUP_EXT) | set(_TABULAR_EXT) | set(_DOC_EXT) | set(_STRUCT_EXT)
)


def _is_binaryish(kind: str) -> bool:
    return kind in ("pdf", "docx", "pptx", "excel", "parquet", "odt", "epub")


def read_text_file(path: Path) -> str:
    """Decode a text file, guessing the encoding when it is not UTF-8."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    try:
        import chardet

        guess = chardet.detect(raw[:200_000])
        if guess.get("encoding"):
            return raw.decode(guess["encoding"], errors="replace")
    except Exception:
        pass
    return raw.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def extract_text(path: Path, kind: str | None = None) -> str:
    """The readable content of a file, whatever container it arrives in."""
    kind = kind or detect_kind(path)
    if kind in ("pdf", "docx", "pptx", "notebook", "odt", "epub"):
        return docs.extract(path, kind)[:MAX_TEXT_CHARS]
    if kind in ("csv", "tsv", "excel", "parquet"):
        return tabular.extract(path, kind)[:MAX_TEXT_CHARS]
    if kind == "html":
        return markup.html_to_text(read_text_file(path))[:MAX_TEXT_CHARS]
    if kind == "latex":
        return latex.to_text(read_text_file(path))[:MAX_TEXT_CHARS]
    return read_text_file(path)[:MAX_TEXT_CHARS]


#: LLM-summary levels are token budgets. None means "no cap".
#: These apply to `summarize_file(mode="llm")` and to compaction, NOT to static
#: outlines -- a deterministic outline has no length to ration.
LLM_LEVEL_TOKENS: dict[int, int | None] = {
    0: 100,
    1: 300,
    2: 1_000,
    3: 3_000,
    4: 10_000,
    5: None,
}
MAX_LEVEL = 5         # deepest structural level; 5 == full document text
MAX_TIER = 4          # deepest per-format *structural* tier
COMPACTION_LEVEL = 4  # LLM budget the compaction engine aims at (10,000 tokens)

#: What each static level adds, for tool descriptions and the UI.
STATIC_LEVELS = {
    0: "barest skeleton (class names, title, shape, top-level keys)",
    1: "+ one more layer (function names, H2, dtypes, depth 2)",
    2: "+ full signatures / deeper headings / per-column statistics",
    3: "+ docstring openings / deeper headings / first rows",
    4: "+ full docstrings / deepest headings / more rows",
    5: "the document's full text",
}


def llm_budget(level: int) -> int | None:
    level = max(0, min(MAX_LEVEL, int(level)))
    return LLM_LEVEL_TOKENS[level]


def describe_llm_levels() -> str:
    parts = []
    for lvl in range(MAX_LEVEL + 1):
        b = LLM_LEVEL_TOKENS[lvl]
        parts.append(f"{lvl}={'uncapped' if b is None else f'{b} tok'}")
    return ", ".join(parts)


def describe_static_levels() -> str:
    return "; ".join(f"{k}={v}" for k, v in STATIC_LEVELS.items())


@functools.lru_cache(maxsize=512)
def _render_tier(path_str: str, stamp: tuple, tier: int, kind: str) -> str:
    """Render one structural level. Cached on (path, mtime, size, level, kind).

    A folder survey outlines the same file at more than one level, and a 200 MB
    CSV must not be re-parsed each time.
    """
    path = Path(path_str)
    return _render_tier_uncached(path, tier, kind)


def _render_tier_uncached(path: Path, level: int, kind: str) -> str:
    if kind == "python":
        body = code.outline_python(read_text_file(path), level)
    elif kind in ("c-like", "ruby", "matlab", "julia", "r", "shell", "sql"):
        body = code.outline_generic(read_text_file(path), level, kind)
    elif kind == "markdown":
        body = markup.outline_markdown(read_text_file(path), level)
    elif kind == "rst":
        body = markup.outline_rst(read_text_file(path), level)
    elif kind == "html":
        body = markup.outline_html(read_text_file(path), level)
    elif kind == "latex":
        body = latex.outline(read_text_file(path), level)
    elif kind in ("csv", "tsv", "excel", "parquet"):
        body = tabular.outline(path, level, kind)
    elif kind in ("pdf", "docx", "pptx", "notebook", "odt", "epub"):
        body = docs.outline(path, level, kind)
    elif kind in ("json", "jsonl", "yaml", "toml", "ini", "xml"):
        body = structured.outline(path, level, kind)
    else:
        body = markup.outline_text(read_text_file(path), level)
    return body[:MAX_OUTLINE_CHARS]


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def outline(path: Path, level: int = 1, kind: str | None = None,
            *, include_header: bool = True) -> str:
    """Deterministic structural summary of `path` at structural `level`.

    The level selects *what kind* of detail to include, not how many tokens to
    spend. A static outline is free to compute and cannot hallucinate, so there
    is nothing to ration -- the size simply follows from the file. (LLM
    summaries are the opposite case, and take a token budget instead; see
    `readers/llm.py`.) Level 5 returns the document's full text.

    `include_header=False` is for callers that print their own label, such as
    the folder overview.
    """
    kind = kind or detect_kind(path)
    level = max(0, min(MAX_LEVEL, int(level)))
    stamp = _stamp(path)

    if level >= MAX_LEVEL:
        try:
            body = extract_text(path, kind)
        except Exception as e:
            body = f"[extraction failed: {type(e).__name__}: {e}]"
        note = "full content"
    else:
        try:
            body = _render_tier(str(path), stamp, level, kind)
        except Exception as e:
            body = f"[outline failed at level {level}: {type(e).__name__}: {e}]"
        note = STATIC_LEVELS[level]

    if not include_header:
        return body[:MAX_OUTLINE_CHARS]
    return (_file_header(path, kind, level, note) + "\n" + body)[:MAX_OUTLINE_CHARS]


def _file_header(path: Path, kind: str, level: int, note: str) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    bits = [f"{path}", f"kind={kind}", f"size={_human(size)}",
            f"level={level} ({note})"]
    if not _is_binaryish(kind):
        try:
            n = read_text_file(path).count("\n") + 1
            bits.insert(2, f"{n} lines")
        except Exception:
            pass
    return "=== " + "  \u00b7  ".join(bits) + " ==="



def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
